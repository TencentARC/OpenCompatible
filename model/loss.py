import torch
from torch import nn, Tensor
from typing import Tuple
from torch.autograd import Variable
import random
import warnings
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Sequence

__all__ = ['BackwardCompatibleLoss']


def gather_tensor(raw_tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensor_large = [torch.zeros_like(raw_tensor) \
                    for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_large, raw_tensor.contiguous())
    tensor_large = torch.cat(tensor_large, dim=0)
    return tensor_large


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(x, y.t(), beta=1, alpha=-2)
    dist_m = dist_m.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist_m


def calculate_loss(feat_new, feat_old, feat_new_large, feat_old_large, \
                   masks, loss_type, temp, criterion, loss_lambda=[1, 1], topk_neg=-1):
    B, D = feat_new.shape
    labels_idx = torch.arange(B) + torch.distributed.get_rank() * B
    if feat_new_large is None:
        feat_new_large = feat_new
        feat_old_large = feat_old

    if (loss_type == 'contra'):
        logits_n2o_pos = torch.bmm(feat_new.view(B, 1, D), feat_old.view(B, D, 1))  # B*1
        logits_n2o_pos = torch.squeeze(logits_n2o_pos, 1)
        logits_n2o_neg = torch.mm(feat_new, feat_old_large.permute(1, 0))  # B*B
        logits_n2o_neg = logits_n2o_neg - masks * 1e9
        if topk_neg > 0:
            logits_n2o_neg = torch.topk(logits_n2o_neg, topk_neg, dim=1)[0]
        logits_all = torch.cat((logits_n2o_pos, logits_n2o_neg), 1)  # B*(1+k)
        logits_all /= temp

        labels_idx = torch.zeros(B).long().cuda()
        loss = criterion(logits_all, labels_idx) * loss_lambda[0]

    elif (loss_type in ['contra_reg_free', 'contra_reg_free_dynamic']):
        logits_n2o_pos = torch.bmm(feat_new.view(B, 1, D), feat_old.view(B, D, 1))  # B*1
        logits_n2o_neg = torch.mm(feat_new, feat_old_large.permute(1, 0))  # B*B
        logits_n2o_neg = logits_n2o_neg - masks * 1e9
        logits_n2n_neg = torch.mm(feat_new, feat_new_large.permute(1, 0))  # B*B
        logits_n2n_neg = logits_n2n_neg - masks * 1e9
        if topk_neg > 0:
            logits_n2o_neg = torch.topk(logits_n2o_neg, topk_neg, dim=1)[0]
            logits_n2n_neg = torch.topk(logits_n2n_neg, topk_neg, dim=1)[0]
        logits_all = torch.cat((logits_n2o_pos, logits_n2o_neg, logits_n2n_neg), 1)  # B*(1+2B)
        logits_all /= temp

        labels_idx = torch.zeros(B).long().cuda()
        loss = criterion(logits_all, labels_idx) * loss_lambda[0]

    elif (loss_type in ['contra_reg_free_v2']):
        logits_n2o_pos = torch.bmm(feat_new.view(B, 1, D), feat_old.view(B, D, 1))  # B*1
        logits_n2o_neg = torch.mm(feat_new, feat_old_large.permute(1, 0))  # B*B
        logits_n2o_neg = logits_n2o_neg - masks * 1e9
        logits_n2n_neg = torch.mm(feat_new, feat_new_large.permute(1, 0))  # B*B
        logits_n2n_neg = logits_n2n_neg - masks * 1e9
        if topk_neg > 0:
            logits_n2o_neg = torch.topk(logits_n2o_neg, topk_neg, dim=1)[0]
            logits_n2n_neg = torch.topk(logits_n2n_neg, topk_neg, dim=1)[0]

        # new2old negative
        logits_all = torch.cat((logits_n2o_pos, logits_n2o_neg), 1)  # B*(1+B)
        logits_all /= temp
        labels_idx = torch.zeros(B).long().cuda()
        loss = criterion(logits_all, labels_idx) * loss_lambda[0]

        # new2new negative
        logits_all = torch.cat((logits_n2o_pos, logits_n2n_neg), 1)  # B*(1+2B)
        logits_all /= temp
        loss += criterion(logits_all, labels_idx) * loss_lambda[1]

    elif (loss_type == 'triplet'):
        logits_n2o = euclidean_dist(feat_new, feat_old_large)
        logits_n2o_pos = torch.gather(logits_n2o, 1, labels_idx.view(-1, 1).cuda())

        # find the hardest negative
        if topk_neg > 0:
            logits_n2o_neg = torch.topk(logits_n2o + masks * 1e9, topk_neg, dim=1, largest=False)[0]

        logits_n2o_pos = logits_n2o_pos.expand_as(logits_n2o_neg).contiguous().view(-1)
        logits_n2o_neg = logits_n2o_neg.view(-1)
        hard_labels_idx = torch.ones_like(logits_n2o_pos)
        loss = criterion(logits_n2o_neg, logits_n2o_pos, hard_labels_idx) * loss_lambda[0]

    elif (loss_type == 'triplet_reg_free'):
        logits_n2o = euclidean_dist(feat_new, feat_old_large)
        logits_n2o_pos = torch.gather(logits_n2o, 1, labels_idx.view(-1, 1).cuda())

        logits_n2n = euclidean_dist(feat_new, feat_new_large)
        # find the hardest negative
        if topk_neg > 0:
            logits_n2o_neg = torch.topk(logits_n2o + masks * 1e9, topk_neg, dim=1, largest=False)[0]
            logits_n2n_neg = torch.topk(logits_n2n + masks * 1e9, topk_neg, dim=1, largest=False)[0]

        logits_n2o_pos = logits_n2o_pos.expand_as(logits_n2o_neg).contiguous().view(-1)
        logits_n2o_neg = logits_n2o_neg.view(-1)
        logits_n2n_neg = logits_n2n_neg.view(-1)
        hard_labels_idx = torch.ones_like(logits_n2o_pos)
        loss = criterion(logits_n2o_neg, logits_n2o_pos, hard_labels_idx) * loss_lambda[0]
        loss += criterion(logits_n2n_neg, logits_n2o_pos, hard_labels_idx) * loss_lambda[1]

    elif (loss_type == 'l2'):
        loss = criterion(feat_new, feat_old) * loss_lambda[0]

    else:
        loss = 0.

    return loss


class BackwardCompatibleLoss(nn.Module):
    def __init__(self, temp=0.05, margin=0.8, topk_neg=-1, loss_type='contra', loss_lambda=[1.0], gather_all=True):
        super(BackwardCompatibleLoss, self).__init__()
        self.temperature = temp
        self.loss_lambda = loss_lambda
        self.topk_neg = topk_neg
        if (loss_type in ['contra', 'contra_reg_free', 'contra_reg_free_v2']):
            self.criterion = nn.CrossEntropyLoss().cuda()
        elif (loss_type == ['contra_reg_free_dynamic']):
            self.criterion = nn.CrossEntropyLoss(reduction='none').cuda()
        elif (loss_type in ['triplet', 'triplet_reg_free']):
            assert topk_neg > 0, \
                "Please select top-k negatives for triplet loss"
            # not use nn.TripletMarginLoss()
            self.criterion = nn.MarginRankingLoss(margin=margin).cuda()
        elif (loss_type == 'l2'):
            self.criterion = nn.MSELoss().cuda()
        else:
            raise NotImplementedError("Unknown loss type: {}".format(loss_type))
        self.loss_type = loss_type
        self.gather_all = gather_all

    def forward(self, inputs, inputs_old, targets):
        # features l2-norm
        z = F.normalize(inputs, dim=1, p=2)
        z_old = F.normalize(inputs_old, dim=1, p=2).detach()
        batch_size = z.size(0)

        # gather tensors from all GPUs
        if self.gather_all:
            z_large = gather_tensor(z)
            z_old_large = gather_tensor(z_old)
            targets_large = gather_tensor(targets)
            batch_size_large = z_large.size(0)
            current_gpu = dist.get_rank()
            masks = targets_large.expand(batch_size_large, batch_size_large) \
                .eq(targets_large.expand(batch_size_large, batch_size_large).t())
            masks = masks[current_gpu * batch_size: (current_gpu + 1) * batch_size, :]  # size: (B, B*n_gpus)
        else:
            z_large, z_old_large = None, None
            masks = targets.expand(batch_size, batch_size).eq(targets.expand(batch_size, batch_size).t())

        # compute loss
        loss_comp = calculate_loss(z, z_old, z_large, z_old_large, masks, self.loss_type, self.temperature,
                                   self.criterion, self.loss_lambda, self.topk_neg)
        return loss_comp

    def forward_with_gradient(self, inputs, inputs_old, targets):
        if not self.no_norm:
            # features l2-norm
            z = F.normalize(inputs, dim=1, p=2)
            z_old = F.normalize(inputs_old, dim=1, p=2)
        else:
            z = inputs
            z_old = inputs_old
        z_large = gather_tensor(z)
        z_old_large = gather_tensor(z_old)
        targets_large = gather_tensor(targets)

        # compute loss
        loss_comp = calculate_loss(z_large, z_old_large, targets_large, self.loss_type, self.temperature,
                                   self.criterion, self.topk_neg)
        return loss_comp


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
