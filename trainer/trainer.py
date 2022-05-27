import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseTrainer
from evaluate.evaluate import evaluate_func
from model.margin_softmax import large_margin_module
from utils import MetricTracker
from utils.util import AverageMeter, tensor_to_float


class LandmarkTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, backward_compatible_training, train_loader, criterion, optimizer, grad_scaler, config, device,
                 validation_loader_list=[None, None, None], test_loader_list=[None, None, None], lr_scheduler=None):
        super().__init__(model, criterion, optimizer, config)
        self.config = config
        self.device = device
        self.grad_scaler = grad_scaler
        self.len_epoch = len(self.train_loader)

        self.backward_compatible_training = backward_compatible_training
        self.train_loader = train_loader
        self.query_loader_public, self.gallery_loader_public, self.query_gts_public = validation_loader_list
        self.query_loader_private, self.gallery_loader_private, self.query_gts_private = test_loader_list

        self.do_validation = self.query_loader is not None
        self.lr_scheduler = lr_scheduler

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs):
            if self.config.distributed:
                self.train_loader.sampler.set_epoch(epoch)

            epoch_time = time.time()
            if self.backward_compatible_training:
                self._back_comp_train_epoch(epoch)
            else:
                self._train_epoch(epoch)
            epoch_time = time.time() - epoch_time
            self.logger.info(f"Epoch {epoch + 1} training takes {epoch_time / 60.0:.2f} minutes")

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            if (epoch + 1) % self.config.val_freq == 0:
                acc1 = evaluate_func(self.model, self.query_loader_public, self.gallery_loader_public,
                                     self.query_gts_public, self.logger, self.config)
                best_acc1 = max(acc1, best_acc1)
                self.logger.info(f"best acc: {best_acc1:.4f}")

            if self.config.rank == 0 and (epoch + 1) % self.save_period == 0:
                self.logger.info('==> Saving checkpoint')
                self._save_checkpoint(epoch, best_acc1)

            if (epoch + 1) == self.config.trainer.epochs:
                self.logger.info(f"=> Test performance in img size {self.config.test_dataset.img_size}:")
                evaluate_func(self.model, self.query_loader_private, self.gallery_loader_private,
                              self.query_gts_private, self.logger, self.config)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        batch_time = AverageMeter('BatchTime', ':6.3f')
        data_time = AverageMeter('DataTime', ':6.3f')
        losses = AverageMeter('Loss', ':.4f')
        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, data_time, losses],
            prefix=f"Epoch:[{epoch + 1}/{self.config.epochs}]  ", logger=self.logger,
        )

        self.model.train()
        self.train_metrics.reset()
        # num_step_in_epoch = len(self.train_loader)
        end_time = time.time()
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end_time)

            # compute output
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                images, labels = images.to(self.device), labels.to(self.device)
                if self.config.loss.type is "softmax":
                    feat, cls_score = self.model(images.cuda())
                else:
                    feat, cls_score = self.model(images.cuda(), use_margin=True)
                    cls_score = large_margin_module(self.config.loss.type, cls_score, labels,
                                                    s=self.config.loss.scale,
                                                    m=self.config.loss.margin)
                loss = self.criterion['base'](cls_score, labels)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            losses.update(loss.item(), images.size(0))
            # grad_scaler can handle the case that use_amp=False indicated in the official pytorch doc
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % self.config.trainer.print_period == 0:
                progress.display(batch_idx, suffix=f"\tlr:{self.optimizer.param_groups[0]['lr']:.6f}")

            self.lr_scheduler.step_update(epoch * self.len_epoch + batch_idx + 1)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()


    def _back_comp_train_epoch(self, epoch):
        batch_time = AverageMeter('BatchTime', ':6.3f')
        data_time = AverageMeter('DataTime', ':6.3f')
        losses_cls = AverageMeter('Loss', ':.4f')
        losses_back_comp_p2c = AverageMeter('Point2Center Backward Compatible Loss', ':.4e')
        losses_back_comp_p2p = AverageMeter('Point2Point Backward Compatible Loss', ':.4e')
        losses_all = AverageMeter('Total Loss', ':.4e')
        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, data_time, losses_all, losses_cls, losses_back_comp_p2c, losses_back_comp_p2p,],
            prefix=f"Epoch:[{epoch + 1}/{self.config.epochs}]  ", logger=self.logger,
        )

        self.model.train()
        self.train_metrics.reset()
        # num_step_in_epoch = len(self.train_loader)
        end_time = time.time()
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end_time)

            # compute output
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                images, labels = images.to(self.device), labels.to(self.device)
                if self.config.loss.type is "softmax":
                    feat, cls_score = self.model.new_model(images.cuda())
                    old_feat, _ = self.model.old_model(images.cuda())
                else:
                    feat, cls_score = self.model.new_model(images.cuda(), use_margin=True)
                    old_feat, _ = self.model.old_model(images.cuda(), use_margin=True)
                    cls_score = large_margin_module(self.config.loss.type, cls_score, labels,
                                                    s=self.config.loss.scale,
                                                    m=self.config.loss.margin)
                loss = self.criterion['base'](cls_score, labels)

                n2o_cls_score = self.model.old_classifier(feat)

                # Point2center backward compatible loss
                if self.config.back_comp_p2c:
                    loss_back_comp_p2c = self.criterion['ce'](n2o_cls_score, labels)

                # Point2point backward compatible loss
                if self.config.back_comp_p2p:
                    if self.config.back_comp_type == 'lwf':
                        pseudo_score = self.model.old_classifier.module(old_feat)
                        pseudo_label = F.softmax(pseudo_score / self.config.distillation_temp, dim=1)
                        loss_back_comp_p2p = -torch.sum(
                            F.log_softmax(n2o_cls_score / self.config.distillation_temp) * pseudo_label) / images.size(
                            0)
                    elif self.config.back_comp_type == 'focal_distillation':
                        criterion_mse = nn.MSELoss(reduce=False).cuda(self.config.device)
                        loss_back_comp_p2p = torch.mean(criterion_mse(feat, old_feat), dim=1)
                        predicted_target = cls_score.argmax(dim=1)
                        bool_target_is_match = (predicted_target == labels)
                        focal_weight = self.config.focal_beta * bool_target_is_match + self.config.focal_alpha
                        loss_back_comp_p2p = torch.mul(loss_back_comp_p2p, focal_weight).mean()
                    elif self.config.back_comp_type in ['contra', 'contra_reg_free', 'contra_reg_free_v2', 'triplet',
                                                 'triplet_reg_free', 'l2']:
                        loss_back_comp_p2p = self.criterion['back_comp'](feat, old_feat, labels)
                    else:
                        raise NotImplementedError("Unknown loss type")
                else:
                    loss_back_comp_p2p = 0.

            losses_cls.update(loss.item(), images.size(0))
            loss_all = loss

            if self.config.back_comp_p2c:
                loss_back_comp_p2c_value = tensor_to_float(loss_back_comp_p2c)
                losses_back_comp_p2c.update(loss_back_comp_p2c_value, len(labels))
                loss_all += loss_back_comp_p2c
            if self.config.back_comp_p2p:
                loss_back_comp_p2p_value = tensor_to_float(loss_back_comp_p2p)
                losses_back_comp_p2p.update(loss_back_comp_p2p_value, len(labels))
                loss_all += loss_back_comp_p2p

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            losses_all.update(loss_all.item(), images.size(0))
            # grad_scaler can handle the case that use_amp=False indicated in the official pytorch doc
            self.grad_scaler.scale(loss_all).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % self.config.trainer.print_period == 0:
                progress.display(batch_idx, suffix=f"\tlr:{self.optimizer.param_groups[0]['lr']:.6f}")

            self.lr_scheduler.step_update(epoch * self.len_epoch + batch_idx + 1)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

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
