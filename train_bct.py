import argparse
import collections
import os
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn

import data_loader.data_loaders as module_data
from model.build import build_model, build_lr_scheduler
from parse_config import ConfigParser
from trainer import LandmarkTrainer, FaceTrainer
from utils import parser_option, cudalize, set_random_seed, resume_checkpoint
from model.loss import BackwardCompatibleLoss


def main(config):
    # fix random seeds for reproducibility
    if config.seed is not None:
        set_random_seed(config.seed)

    if config.dist_url == "env://" and config.world_size == -1:
        config.world_size = int(os.environ["WORLD_SIZE"])

    config.distributed = config.world_size > 1 or config.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if config.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config.world_size = ngpus_per_node * config.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config, None))
    else:
        # Simply call main_worker function
        main_worker(config.device, ngpus_per_node, config)


def main_worker(device, ngpus_per_node, config):
    if config.distributed:
        if config.dist_url == "env://" and config.rank == -1:
            config.rank = int(os.environ["RANK"])
        if config.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            config.rank = config.rank * ngpus_per_node + device
        dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url, \
                                world_size=config.world_size, rank=config.rank)
        dist.barrier()
    cudnn.benchmark = True

    if not config.multiprocessing_distributed or \
            (config.multiprocessing_distributed and torch.distributed.get_rank() == 0):
        is_print = True
    else:
        is_print = False
    config["is_print"] = is_print

    logger = config.get_logger('train')

    if config.dataset.type == 'landmark':
        # load training set
        data_loader = module_data.GLDv2TrainDataLoader(config)
        train_loader, class_num = data_loader.train_loader, data_loader.class_num
        # load evaluation set
        data_loader = module_data.GLDv2EvalDataLoader(config)
        eval_query_loader, eval_gallery_loader, eval_query_gts = data_loader.query_loader, \
                                                                 data_loader.gallery_loader, data_loader.query_gts
        # load test set
        data_loader = module_data.GLDv2TestDataLoader(config)
        test_query_loader, test_gallery_loader, test_query_gts = data_loader.query_loader, \
                                                                 data_loader.gallery_loader, data_loader.query_gts

    elif config.dataset.type == 'face':
        data_loader = module_data.MS1Mv3TrainDataLoader(config)
    else:
        raise NotImplementedError

    # build model architecture
    model = build_model(config)


    # build solver
    optimizer = torch.optim.SGD(model.parameters(), lr=config.optimizer.lr,
                                momentum=config.optimizer.momentum,
                                weight_decay=config.optimizer.weight_decay)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=config.trainer.use_amp)

    best_acc1 = 0.

    if config.new_model.resume is not None:
        best_acc1 = resume_checkpoint(model.new_model.module, optimizer, grad_scaler, config, logger)
    model.new_model = cudalize(model.new_model, config)
    model.old_model = cudalize(model.old_model, config)
    model.old_classifier = cudalize(model.old_classifier, config)

    steps_per_epoch = len(train_loader)
    lr_scheduler = build_lr_scheduler(config, optimizer, steps_per_epoch, sche_type=config.lr_scheduler.type)


    # build loss
    criterion = {}
    criterion['base'] = nn.CrossEntropyLoss().cuda()
    criterion['back_comp'] = BackwardCompatibleLoss(temp=config.temp, margin=config.triplet_margin, \
                                                    topk_neg=config.topk_neg, \
                                                    loss_type='contra', loss_lambda=config.loss_lambda,  \
                                                    gather_all=config.gather_all)
    if config.dataset.type == 'landmark':
        trainer = LandmarkTrainer(model,
                                  backward_compatible_training=True, \
                                  train_loader=train_loader, \
                                  criterion=criterion, \
                                  optimizer=optimizer, \
                                  grad_scaler=grad_scaler, \
                                  config=config, \
                                  device=device, \
                                  validation_loader_list=[eval_query_loader, eval_gallery_loader, eval_query_gts], \
                                  test_loader_list=[test_query_loader, test_gallery_loader, test_query_gts], \
                                  lr_scheduler=lr_scheduler)
    elif config.dataset_type == 'face':
        trainer = FaceTrainer()
    else:
        raise NotImplementedError

    trainer.train()


if __name__ == '__main__':
    config = parser_option()
    main(config)
