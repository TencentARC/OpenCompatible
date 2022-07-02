import torch
import torch.nn as nn
import torch.nn.functional as F


class Resnet_GeM(nn.Module):
    def __init__(self, backbone, num_classes, emb_dim):
        super().__init__()
        self.backbone = backbone
        self.backbone.avgpool = GeM()
        self.fc_emb = nn.Linear(self.backbone.fc.in_features, emb_dim)
        self.fc_classify = nn.Linear(emb_dim, num_classes, bias=False)
        self.backbone.fc = nn.Identity()

    def forward(self, x, use_cosine_cla_score=False):
        x = self.backbone(x)
        x = self.fc_emb(x)
        if use_cosine_cla_score:
            cla_score = F.linear(F.normalize(x), F.normalize(self.fc_classify.weight))
        else:
            cla_score = self.fc_classify(x)

        return x, cla_score


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6, freeze_p=True):
        super(GeM, self).__init__()
        self.p = p if freeze_p else torch.nn.parameter.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                                     (1, 1)).pow(1. / self.p)
        # return F.lp_pool2d(F.threshold(x, eps, eps), p, (1, 1))  # alternative
