'''
Modified from https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/losses.py
'''

import torch


def large_margin_module(name, cosine, label, s, m):
    if name == "arcface":
        return arcface(cosine, label, s, m)
    elif name == "cosface":
        return cosface(cosine, label, s, m)
    else:
        raise NotImplementedError


def cosface(cosine, label, s, m):
    index = torch.where(label != -1)[0]
    m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
    m_hot.scatter_(1, label[index, None], m)
    cosine[index] -= m_hot
    ret = cosine * s
    return ret


def arcface(cosine, label, s, m):
    index = torch.where(label != -1)[0]
    m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
    m_hot.scatter_(1, label[index, None], m)
    cosine.acos_()
    cosine[index] += m_hot
    cosine.cos_().mul_(s)
    return cosine
