import os
from abc import abstractmethod

import torch

from logger import TensorboardWriter


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, train_loader, criterion, optimizer, grad_scaler, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

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

        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch, best_acc1):
        """
        Saving checkpoints

        :param epoch: current epoch number
        """
        arch = type(self.model).__name__
        checkpoint = {
            'epoch': epoch + 1,
            'arch': arch,
            'model': self.model.module.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': self.optimizer.state_dict(),
            'grad_scaler': self.grad_scaler.state_dict(),
            'config': self.config
        }
        ckpt_path = os.path.join(self.checkpoint_dir, "ckpt", f'checkpoint_epoch{epoch + 1}.pth.tar')
        torch.save(checkpoint, ckpt_path)
        self.logger.info("Saving checkpoint: {} ...".format(ckpt_path))

    # TODO
    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
