import argparse
import os
from datetime import timedelta

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn

import data_loader.data_loaders as module_data
from model.build import build_model, build_lr_scheduler
from trainer import LandmarkTrainer, FaceTrainer
from logger.logger import create_logger
from utils.util import parser_option, cudalize, set_random_seed, resume_checkpoint
from evaluate.evaluate import evaluate_func


def main(config):
    args = argparse.Namespace(**config.config)

    # fix random seeds for reproducibility
    if args.seed is not None:
        set_random_seed(args.seed)

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    config._update_config_by_dict({"distributed": args.distributed})
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, config))
    else:
        main_worker(args.device, ngpus_per_node, config)


def main_worker(device, ngpus_per_node, args, config):
    args.device = device
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + device
            config._update_config_by_dict({"rank": args.rank})
            config._update_config_by_dict({"device": args.device})
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank, timeout=timedelta(seconds=45))
        # dist.barrier()
    cudnn.benchmark = True

    logger = create_logger(output_dir=config.log_dir, dist_rank=args.rank, name='eval')

    if args.test_dataset["type"] == 'landmark':
        # load test set
        if args.test_dataset["name"] == 'gldv2':
            data_loader = module_data.GLDv2TestDataset(args)
        elif args.test_dataset["name"] in ['roxford', 'rparis']:
            data_loader = module_data.ROxfordTestDataLoader(args)
        else:
            raise NotImplementedError
        query_loader = data_loader.query_loader
        gallery_loader = data_loader.gallery_loader
        query_gts = data_loader.query_gts
    elif args.test_dataset["type"] == 'face':
        # data_loader = module_data.IJBCTestDataLoader(args)
        pass
    else:
        raise NotImplementedError

    # build model architecture
    model = build_model(args, logger)

    if args.old_model["arch"] is None:
        model = cudalize(model, args)
        evaluate_func(model, query_loader, gallery_loader, query_gts, logger, config,
                      old_model=None, dataset_name=args.test_dataset["name"])
    else:
        model.new_model = cudalize(model.new_model, args)
        model.old_model.cuda(args.device)
        logger.info(f"=> Self-model performance:")
        evaluate_func(model.new_model, query_loader, gallery_loader, query_gts, logger, config,
                      old_model=None, dataset_name=args.test_dataset["name"])

        logger.info(f"=> Cross-model performance:")
        evaluate_func(model.new_model, query_loader, gallery_loader, query_gts, logger, config,
                      old_model=model.old_model, dataset_name=args.test_dataset["name"])
    return


if __name__ == '__main__':
    config = parser_option()
    main(config)
