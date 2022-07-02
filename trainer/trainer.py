import os
from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from evaluate.evaluate import evaluate_func
from model.margin_softmax import large_margin_module
from utils.util import AverageMeter, tensor_to_float
from torch.utils.tensorboard import SummaryWriter


class LandmarkTrainer:
    """
    Trainer class
    """

    def __init__(self, model, comp_training, train_loader,
                 criterion, optimizer, grad_scaler, args, config, logger,
                 validation_loader_list=[None, None, None],
                 test_loader_list=[None, None, None],
                 lr_scheduler=None):
        self.config = config
        self.logger = logger
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.grad_scaler = grad_scaler

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.start_epoch = 0

        self.checkpoint_dir = config.save_dir
        self.writer = SummaryWriter(config.log_dir)

        self.args = args
        self.best_acc1 = self.config.config["best_acc1"]
        self.device = args.device
        self.len_epoch = len(self.train_loader)

        self.comp_training = comp_training
        self.query_loader_public, self.gallery_loader_public, self.query_gts_public = validation_loader_list
        self.query_loader_private, self.gallery_loader_private, self.query_gts_private = test_loader_list

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

            if self.comp_training is None:
                self._train_epoch(epoch)
            elif self.comp_training == 'backward':
                self._back_comp_train_epoch(epoch)
            elif self.for_comp_training == 'forward':
                self._for_comp_train_epoch(epoch)
            else:
                raise NotImplementedError

            if self.comp_training is None:
                _model = self.model
                _old_model = None
            else:
                _model = self.model.new_model
                _old_model = self.model.old_model

            epoch_time = time.time() - epoch_time

            if not self.args.multiprocessing_distributed or \
                    (self.args.multiprocessing_distributed and torch.distributed.get_rank() == 0):
                self.logger.info(f"Epoch {epoch + 1} training takes {epoch_time / 60.0:.2f} minutes")

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            if (epoch + 1) % self.args.trainer["val_period"] == 0:
                acc1 = evaluate_func(model=_model,
                                     query_loader=self.query_loader_public,
                                     gallery_loader=self.gallery_loader_public,
                                     query_gts=self.query_gts_public,
                                     logger=self.logger,
                                     config=self.config,
                                     old_model=_old_model,
                                     dataset_name=self.args.test_dataset["name"])
                self.best_acc1 = max(acc1, self.best_acc1)
                self.config._update_config_by_dict({"best_acc1": self.best_acc1})
                self.logger.info(f"best acc: {self.best_acc1:.4f}")

            if self.args.rank == 0 and (epoch + 1) >= 5 and (epoch + 1) % self.save_period == 0:
                self.logger.info('==> Saving checkpoint')
                self._save_checkpoint(epoch, _model)

            if (epoch + 1) == self.args.trainer["epochs"]:
                self.logger.info(f"=> Self-model performance in img size={self.args.test_dataset['img_size']}:")
                acc1 = evaluate_func(model=_model,
                                     query_loader=self.query_loader_private,
                                     gallery_loader=self.gallery_loader_private,
                                     query_gts=self.query_gts_private,
                                     logger=self.logger,
                                     config=self.config,
                                     old_model=None,
                                     dataset_name=self.args.test_dataset["name"])
                if self.comp_training is not None:
                    self.logger.info(f"=> Cross-model performance:")
                    acc1 = evaluate_func(model=_model,
                                         query_loader=self.query_loader_private,
                                         gallery_loader=self.gallery_loader_private,
                                         query_gts=self.query_gts_private,
                                         logger=self.logger,
                                         config=self.config,
                                         old_model=_old_model,
                                         dataset_name=self.args.test_dataset["name"])

                self.config._update_config_by_dict({"final_acc1": acc1})


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
        end_time = time.time()
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end_time)

            total_steps = epoch * self.len_epoch + batch_idx

            # compute output
            with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                images, labels = images.to(self.device), labels.to(self.device)
                feat = self.model(images.cuda())
                # loss function options: softmax/arcface/cosface
                if self.args.loss["type"] == "softmax":
                    cls_score = self.model.fc_classifier(feat)
                else:
                    cls_score = F.linear(F.normalize(feat), F.normalize(self.model.fc_classifier.weight))
                    cls_score = large_margin_module(self.args.loss["type"], cls_score, labels,
                                                    s=self.args.loss["scale"],
                                                    m=self.args.loss["margin"])
                loss = self.criterion['base'](cls_score, labels)

            self.writer.add_scalar("Loss", losses.avg, total_steps)

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
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], total_steps)

            self.lr_scheduler.step_update(total_steps + 1)

    def _back_comp_train_epoch(self, epoch):
        batch_time = AverageMeter('BatchTime', ':6.3f')
        data_time = AverageMeter('DataTime', ':6.3f')
        losses_cls = AverageMeter('Cls Loss', ':.4f')
        losses_back_comp = AverageMeter('Backward Comp Loss', ':.4f')
        losses_all = AverageMeter('Total Loss', ':.4e')
        epochs = self.args.trainer["epochs"]
        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, data_time, losses_all, losses_cls, losses_back_comp],
            prefix=f"Epoch:[{epoch + 1}/{epochs}]  ", logger=self.logger,
        )

        self.model.train()
        end_time = time.time()
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end_time)

            total_steps = epoch * self.len_epoch + batch_idx

            # compute output
            with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                images, labels = images.to(self.device), labels.to(self.device)
                feat, feat_old = self.model(images.cuda())
                if self.args.loss["type"] == "softmax":
                    cls_score = self.model.new_model.module.fc_classifier(feat)
                else:
                    cls_score = F.linear(F.normalize(feat), F.normalize(self.model.new_model.module.fc_classifier.weight))
                    cls_score = large_margin_module(self.args.loss["type"], cls_score, labels,
                                                    s=self.args.loss["scale"],
                                                    m=self.args.loss["margin"])
                loss = self.criterion['base'](cls_score, labels)

                n2o_cls_score = self.model.old_model.fc_classifier(feat)

                # Point2center backward compatible loss (original BCT loss),
                # from paper "Towards backward-compatible representation learning", CVPR 2020
                if self.args.comp_loss["type"] == 'bct':
                    loss_back_comp = self.criterion['back_comp'](n2o_cls_score, labels)
                elif self.args.comp_loss["type"] == 'bct_ract':
                    masks = F.one_hot(labels, num_classes=cls_score.size(1))
                    masked_cls_score = cls_score - masks * 1e9
                    concat_cls_score = torch.cat((n2o_cls_score, masked_cls_score), 1)
                    loss_back_comp = self.criterion['back_comp'](concat_cls_score, labels)
                else:
                    # Point2point backward compatible loss
                    # Options:
                    #   - lwf (paper: "Learning without Forgetting")
                    #   - fd (paper: "Positive-congruent training: Towards regression-free model updates", CVPR 2021)
                    #   - contra
                    #   - triplet
                    #   - l2
                    #   - contra_ract (paper: "Hot-Refresh Model Upgrades with Regression-Free Compatible Training in Image Retrieval", ICLR 2022)
                    #   - triplet_ract
                    if self.args.comp_loss["type"] == 'lwf':
                        old_cls_score = self.model.old_model.fc_classifier(feat_old)
                        old_cls_score = F.softmax(old_cls_score / self.args.comp_loss["distillation_temp"], dim=1)
                        loss_back_comp = -torch.sum(
                            F.log_softmax(n2o_cls_score / self.args.comp_loss["temperature"]) * old_cls_score) \
                                         / images.size(0)
                    elif self.args.comp_loss["type"] == 'fd':
                        criterion_mse = nn.MSELoss(reduce=False).cuda(self.args.device)
                        loss_back_comp = torch.mean(criterion_mse(feat, feat_old), dim=1)
                        predicted_target = cls_score.argmax(dim=1)
                        bool_target_is_match = (predicted_target == labels)
                        focal_weight = self.args.comp_loss["focal_beta"] * bool_target_is_match + self.args.comp_loss[
                            "focal_alpha"]
                        loss_back_comp = torch.mul(loss_back_comp, focal_weight).mean()
                    elif self.args.comp_loss["type"] in ['contra', 'triplet', 'l2', 'contra_ract', 'triplet_ract']:
                        loss_back_comp = self.criterion['back_comp'](feat, feat_old, labels)
                    else:
                        raise NotImplementedError("Unknown backward compatible loss type")

            losses_cls.update(loss.item(), images.size(0))
            self.writer.add_scalar("Cls loss", losses_cls.avg, total_steps)

            loss_back_comp_value = tensor_to_float(loss_back_comp)
            losses_back_comp.update(loss_back_comp_value, len(labels))
            loss = loss + loss_back_comp
            self.writer.add_scalar("Comp loss", losses_back_comp.avg, total_steps)


            losses_all.update(loss.item(), images.size(0))
            self.writer.add_scalar("Total loss", losses_all.avg, total_steps)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            # measure elapsed time
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if batch_idx % self.args.trainer["print_period"] == 0:
                progress.display(batch_idx, suffix=f"\tlr:{self.optimizer.param_groups[0]['lr']:.6f}")
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], total_steps)

            self.lr_scheduler.step_update(total_steps + 1)


    def _for_comp_train_epoch(self, epoch):
        """
        Forward Compatible Training
        """
        return

    def _save_checkpoint(self, epoch, _model):
        """
        Saving checkpoints

        :param epoch: current epoch number
        """
        arch = type(self.model).__name__
        checkpoint = {
            'epoch': epoch + 1,
            'arch': arch,
            'model': _model.module.state_dict(),
            'best_acc1': self.best_acc1,
            'optimizer': self.optimizer.state_dict(),
            'grad_scaler': self.grad_scaler.state_dict(),
            'config': self.config
        }
        save_dir = Path(os.path.join(self.checkpoint_dir, "ckpt"))
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / f'checkpoint_epoch{epoch + 1}.pth.tar'
        torch.save(checkpoint, ckpt_path)
        self.logger.info("Saving checkpoint: {} ...".format(ckpt_path))


class FaceTrainer:
    """
    Trainer class
    """

    def __init__(self, model, train_loader, criterion, metric_ftns, optimizer, config, device,
                 valid_loader=None, lr_scheduler=None, len_epoch=None):
        return


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", logger=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch, suffix=''):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.logger.info('\t'.join(entries) + suffix)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return 'Iter:[' + fmt + '/' + fmt.format(num_batches) + ']'
