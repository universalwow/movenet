from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _gather_feat_plus(feat, ind):
    print(f'_gather_feat_plus feat {feat.shape} ind {ind.shape}')
    # _gather_feat_plus feat torch.Size([8, 4096, 1, 2]) ind torch.Size([8, 2])
    # num_objs = ind.size(1) / 17
    # print(f'_gather_feat_plus feat 1 {feat[:, ind[0][0]]} {feat[:, ind[0][1]]}')
    # _gather_feat_plus feat torch.Size([1, 4096, 17, 2]) ind torch.Size([1, 17])
    ind = ind.view(ind.size(0), -1, feat.size(2))
    print(f'_gather_feat_plus ind {ind} ')
    ind = ind.unsqueeze(3).expand(ind.size(0), ind.size(1), ind.size(2), 2)
    # print(f'_gather_feat_plus ind {ind.shape} {ind}')
    # _gather_feat_plus ind torch.Size([8, 2, 1, 2])
    feat = feat.gather(1, ind)
    print(f'_gather_feat_plus feat 2 {feat.shape}')
    # _gather_feat_plus feat 2 torch.Size([1, 2, 1, 2])
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _transpose_and_gather_feat_plus(feat, ind):
    print(f'_transpose_and_gather_feat_plus feat {feat.shape}')
    # _transpose_and_gather_feat_plus feat torch.Size([8, 2, 64, 64])
    feat = feat.permute(0, 2, 3, 1).contiguous()
    print(f'_transpose_and_gather_feat_plus feat {feat.shape}')
    # _transpose_and_gather_feat_plus feat torch.Size([8, 64, 64, 2])
    feat = feat.view(feat.size(0), -1, feat.size(3)//2, 2)
    # 1 64 * 64, 17 , 2
    feat = _gather_feat_plus(feat, ind)
    feat = feat.view(feat.size(0), -1, 2)
    return feat


def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)


def flip_lr(x, flip_idx):
    tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    shape = tmp.shape
    for e in flip_idx:
        tmp[:, e[0], ...], tmp[:, e[1], ...] = \
            tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
    return torch.from_numpy(tmp.reshape(shape)).to(x.device)


def flip_lr_off(x, flip_idx):
    tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    shape = tmp.shape
    tmp = tmp.reshape(tmp.shape[0], 17, 2,
                      tmp.shape[2], tmp.shape[3])
    tmp[:, :, 0, :, :] *= -1
    for e in flip_idx:
        tmp[:, e[0], ...], tmp[:, e[1], ...] = \
            tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
    return torch.from_numpy(tmp.reshape(shape)).to(x.device)
