import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel


class Resnet_GeM(nn.Module):
    def __init__(self, backbone, class_num, embedding_size):
        super().__init__()
        self.backbone = backbone
        self.backbone.avgpool = GeM()
        self.fc_emb = nn.Linear(self.backbone.fc.in_features, embedding_size)
        self.fc_classifier = nn.Linear(embedding_size, class_num, bias=False)
        self.backbone.fc = nn.Identity()

    def forward(self, x, use_margin=False):
        x = self.backbone(x)
        x = self.fc_emb(x)
        if use_margin:
            cls_score = F.linear(F.normalize(x), F.normalize(self.fc_classifier.weight))
        else:
            cls_score = self.fc_classifier(x)

        return x, cls_score


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6, freeze_p=True):
        super(GeM, self).__init__()
        self.p = p if freeze_p else torch.nn.parameter.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                                     (1, 1)).pow(1. / self.p)
        # return F.lp_pool2d(F.threshold(x, eps, eps), p, (1, 1))  # alternative


class BackwardCompatibleModel(BaseModel):
    def __init__(self, old_model, new_model, old_classifier=None):
        super(BackwardCompatibleModel, self).__init__()
        self.old_model = old_model
        self.new_model = new_model
        if old_classifier is None:
            self.old_classifier = self.old_model.fc_classifier
        else:
            self.old_classifier = old_classifier

        for param in [self.old_model.parameters(), self.old_classifier.parameters()]:  # fix old parameters
            param.requires_grad = False

        self.training = True

    def train(self):
        self.old_model.eval()  # switch to eval mode during the whole period
        self.old_classifier.eval()
        self.new_model.train()
        self.training = True

    def eval(self):
        self.new_model.eval()
        self.training = False

    def forward(self, x_old, x_new):
        old_feat = self.old_model(x_old).detach()
        new_feat = self.new_model(x_new)
        return old_feat, new_feat
