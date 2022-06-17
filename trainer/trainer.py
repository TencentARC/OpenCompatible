import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseTrainer
from evaluate.evaluate import evaluate_func
from model.margin_softmax import large_margin_module
from utils.util import AverageMeter, tensor_to_float


class LandmarkTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, back_comp_training, for_comp_training, train_loader,
                 criterion, optimizer, grad_scaler, config, device,
                 validation_loader_list=[None, None, None],
                 test_loader_list=[None, None, None],
                 lr_scheduler=None):
        super().__init__(model, train_loader, criterion, optimizer, grad_scaler, config)
        self.args = argparse.Namespace(**config.config)
        self.best_acc1 = self.args.best_acc1
        self.device = device
        self.len_epoch = len(self.train_loader)

        self.back_comp_training = back_comp_training
        self.for_comp_training = for_comp_training
        self.query_loader_public, self.gallery_loader_public, self.query_gts_public = validation_loader_list
        self.query_loader_private, self.gallery_loader_private, self.query_gts_private = test_loader_list

        self.do_validation = self.query_loader_public is not None
        self.lr_scheduler = lr_scheduler


    def train(self):
        """
        Full training logic
        """
        if self.start_epoch > 0:
            num_step_in_epoch = len(self.train_loader)
            self.lr_scheduler.step_update(self.start_epoch * num_step_in_epoch)

        for epoch in range(self.start_epoch, self.epochs):
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)

            epoch_time = time.time()
            if self.back_comp_training:
                self._back_comp_train_epoch(epoch)
            elif self.for_comp_training:
                self._for_comp_train_epoch(epoch)
            else:
                self._train_epoch(epoch)
            epoch_time = time.time() - epoch_time

            if not self.args.multiprocessing_distributed or \
                    (self.args.multiprocessing_distributed and torch.distributed.get_rank() == 0):
                self.logger.info(f"Epoch {epoch + 1} training takes {epoch_time / 60.0:.2f} minutes")

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            if (epoch + 1) % self.args.trainer["val_period"] == 0:
                acc1 = evaluate_func(self.model, self.query_loader_public, self.gallery_loader_public,
                                     self.query_gts_public, self.logger, self.config)
                self.best_acc1 = max(acc1, self.best_acc1)
                self.logger.info(f"best acc: {self.best_acc1:.4f}")

            if self.args.rank == 0 and (epoch + 1) % self.save_period == 0:
                self.logger.info('==> Saving checkpoint')
                self._save_checkpoint(epoch)

            if (epoch + 1) == self.args.trainer["epochs"]:
                self.logger.info(f"=> Test performance in img size {self.args.test_dataset.img_size}:")
                evaluate_func(self.model, self.query_loader_private, self.gallery_loader_private,
                              self.query_gts_private, self.logger, self.args)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        """
        batch_time = AverageMeter('BatchTime', ':6.3f')
        data_time = AverageMeter('DataTime', ':6.3f')
        losses = AverageMeter('Loss', ':.4f')
        epochs = self.args.trainer["epochs"]
        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, data_time, losses],
            prefix=f"Epoch:[{epoch + 1}/{epochs}]  ", logger=self.logger,
        )

        self.model.train()
        # num_step_in_epoch = len(self.train_loader)
        end_time = time.time()
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end_time)

            # compute output
            with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                images, labels = images.to(self.device), labels.to(self.device)
                # loss function options: softmax/arcface/cosface
                if self.args.loss["type"] == "softmax":
                    _, cls_score = self.model(images.cuda())
                else:
                    _, cls_score = self.model(images.cuda(), use_margin=True)
                    cls_score = large_margin_module(self.args.loss["type"], cls_score, labels,
                                                    s=self.args.loss["scale"],
                                                    m=self.args.loss["margin"])
                loss = self.criterion['base'](cls_score, labels)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            losses.update(loss.item(), images.size(0))
            # grad_scaler can handle the case that use_amp=False indicated in the official pytorch doc
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            # measure elapsed time
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if batch_idx % self.args.trainer["print_period"] == 0:
                progress.display(batch_idx, suffix=f"\tlr:{self.optimizer.param_groups[0]['lr']:.6f}")

            self.lr_scheduler.step_update(epoch * self.len_epoch + batch_idx + 1)



    def _back_comp_train_epoch(self, epoch):
        batch_time = AverageMeter('BatchTime', ':6.3f')
        data_time = AverageMeter('DataTime', ':6.3f')
        losses_cls = AverageMeter('Loss', ':.4f')
        losses_back_comp = AverageMeter('Backward Compatible Loss', ':.4f')
        losses_all = AverageMeter('Total Loss', ':.4e')
        epochs = self.args.trainer["epochs"]
        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, data_time, losses_all, losses_cls, losses_back_comp],
            prefix=f"Epoch:[{epoch + 1}/{epochs}]  ", logger=self.logger,
        )

        self.model.new_model.train()
        end_time = time.time()
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end_time)

            # compute output
            with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                images, labels = images.to(self.device), labels.to(self.device)
                if self.args.loss["type"] == "softmax":
                    feat, cls_score = self.model.new_model(images.cuda())
                    old_feat, _ = self.model.old_model(images.cuda())
                else:
                    feat, cls_score = self.model.new_model(images.cuda(), use_margin=True)
                    old_feat, _ = self.model.old_model(images.cuda(), use_margin=True)
                    cls_score = large_margin_module(self.args.loss["type"], cls_score, labels,
                                                    s=self.args.loss["scale"],
                                                    m=self.args.loss["margin"])
                loss = self.criterion['base'](cls_score, labels)

                n2o_cls_score = self.model.old_classifier(feat)

                # Point2center backward compatible loss (original BCT loss), from paper "Towards backward-compatible representation learning"
                if self.args.comp_loss["type"] == 'bct':
                    loss_back_comp = self.criterion['ce'](n2o_cls_score, labels)
                else:
                    # Point2point backward compatible loss
                    # Options:
                    #   - lwf (paper: "")
                    #   - fd (paper: "")
                    #   - contra
                    #   - triplet
                    #   - l2
                    #   - ract (paper: "")
                    if self.args.comp_loss["type"] == 'lwf':
                        old_cls_score = self.model.old_classifier.module(old_feat)
                        old_cls_score = F.softmax(old_cls_score / self.args.comp_loss["distillation_temp"], dim=1)
                        loss_back_comp = -torch.sum(
                            F.log_softmax(n2o_cls_score / self.args.comp_loss["temperature"]) * old_cls_score) / images.size(0)
                    elif self.args.comp_loss["type"] == 'fd':
                        criterion_mse = nn.MSELoss(reduce=False).cuda(self.args.device)
                        loss_back_comp = torch.mean(criterion_mse(feat, old_feat), dim=1)
                        predicted_target = cls_score.argmax(dim=1)
                        bool_target_is_match = (predicted_target == labels)
                        focal_weight = self.args.comp_loss["focal_beta"] * bool_target_is_match + self.args.comp_loss["focal_alpha"]
                        loss_back_comp = torch.mul(loss_back_comp, focal_weight).mean()
                    elif self.args.comp_loss["type"] in ['contra', 'triplet', 'l2', 'ract']:
                        loss_back_comp = self.criterion['back_comp'](feat, old_feat, labels)
                    else:
                        raise NotImplementedError("Unknown loss type")

            losses_cls.update(loss.item(), images.size(0))
            loss_all = loss

            loss_back_comp_value = tensor_to_float(loss_back_comp)
            losses_back_comp.update(loss_back_comp_value, len(labels))
            loss_all += loss_back_comp

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            losses_all.update(loss_all.item(), images.size(0))
            self.grad_scaler.scale(loss_all).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % self.args.trainer["print_period"] == 0:
                progress.display(batch_idx, suffix=f"\tlr:{self.optimizer.param_groups[0]['lr']:.6f}")

            self.lr_scheduler.step_update(epoch * self.len_epoch + batch_idx + 1)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def _for_comp_train_epoch(self, epoch):
        """
        Forward Compatible Training
        """
        pass

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_loader, 'n_samples'):
            current = batch_idx * self.train_loader.batch_size
            total = self.train_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


class FaceTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, train_loader, criterion, metric_ftns, optimizer, config, device,
                 valid_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", logger=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch, suffix=''):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        # print('\t'.join(entries))
        self.logger.info('\t'.join(entries) + suffix)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return 'Iter:[' + fmt + '/' + fmt.format(num_batches) + ']'
