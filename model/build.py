import torch
import torch.nn as nn
import torchvision
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler

from utils.util import load_pretrained_model
from .model import Resnet_GeM, BackwardCompatibleModel
from .inception import Inception3


def build_backbone(model_type="resnet50", class_num=1000, emb_dim=512):
    if model_type == "resnet50":
        backbone = torchvision.models.resnet50()
    elif model_type == "resnet101":
        backbone = torchvision.models.resnet101()
    elif model_type == "inception_v3":
        return Inception3(class_num, emb_dim)
    else:
        raise NotImplementedError
    return Resnet_GeM(backbone, class_num, emb_dim)


def build_classifier(in_dim, out_dim, pretrained_weights):
    assert pretrained_weights is not None
    classifier = nn.Linear(in_dim, out_dim)
    classifier.load_state_dict(torch.load(pretrained_weights))
    return classifier


def build_model(args, logger):
    new_model = build_backbone(model_type=args.new_model["arch"],
                               class_num=args.dataset["class_num"],
                               emb_dim=args.new_model["emb_dim"])
    if args.new_model["pretrained_model_path"]:
        load_pretrained_model(new_model,
                              pretrained_model_path=args.new_model["pretrained_model_path"],
                              model_key_in_ckpt=args.new_model["model_key_in_ckpt"],
                              logger=logger)

    # load old model to train new compatible model
    if args.old_model["arch"] is not None:
        old_model = build_backbone(model_type=args.old_model["arch"],
                                   class_num=args.dataset["class_num"],
                                   emb_dim=args.old_model["emb_dim"])
        load_pretrained_model(old_model,
                              pretrained_model_path=args.old_model["pretrained_model_path"],
                              model_key_in_ckpt=args.old_model["model_key_in_ckpt"],
                              logger=logger)

        # if args.old_model["pretrained_classfier_path"] is not None:
        #     old_classfier = build_classifier(in_dim=args.old_model["emb_dim"], out_dim=args.dataset["class_num"])
        # else:
        #     old_classfier = None

        back_comp_model = BackwardCompatibleModel(old_model, new_model)
        return back_comp_model
    else:
        return new_model


def build_lr_scheduler(args, optimizer, steps_per_epoch, sche_type='cosine'):
    if sche_type == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=args.trainer["epochs"] * steps_per_epoch,
            lr_min=args.optimizer["lr"] * 1e-2,
            warmup_lr_init=args.optimizer["lr"] * 1e-3,
            warmup_t=1 * steps_per_epoch,
            cycle_limit=1,
            t_in_epochs=False,
        )
    elif sche_type == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=args.lr_scheduler["lr_adjust_interval"] * steps_per_epoch,
            decay_rate=0.1,
            warmup_lr_init=args.optimizer["lr"] * 1e-3,
            warmup_t=1 * steps_per_epoch,
            t_in_epochs=False,
        )
    else:
        print(f"{sche_type} not supported.")
        raise NotImplementedError

    return lr_scheduler
