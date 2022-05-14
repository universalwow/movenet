from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pickle import NONE
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from .utils import _gather_feat, _transpose_and_gather_feat, _transpose_and_gather_feat_plus
from .networks.movenet import  joints_name


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk_channel(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs


def _topk_channel_with_reg_kps(scores, reg_kps, K=40, delta=1.0, multipiler=5.0):
    '''
    This function is similar to `_topk`, expect the scroes are weighted by the inverse-distance from the frame center. As the pure inserse function goes down too fast, I add two hyper-parameters here: `delta` and `multiplier`. It's not clear how to tune these two parameters.
    '''
    batch, cat, height, width = scores.size()

    weight_to_joints = torch.zeros_like(scores)
    joint_x = reg_kps[0, :, 0, 0, 0]
    joint_y = reg_kps[0, :, 0, 0, 1]
    # print(joint_x.size())
    # print(joint_y.size())

    y, x = np.ogrid[0:height, 0:width]
    y = np.repeat(np.expand_dims(y, axis=0), cat, axis=0)
    x = np.repeat(np.expand_dims(x, axis=0), cat, axis=0)
    y = y - joint_y.reshape((cat, 1, -1)).cpu().numpy()
    x = x - joint_x.reshape((cat, 1, -1)).cpu().numpy()
    weight_to_joints = torch.from_numpy(multipiler / (np.sqrt(x * x + y * y) + delta)).to(scores.device)
    weight_to_joints = weight_to_joints.reshape((scores.size()))

    scores *= weight_to_joints
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    print(f'_topk {topk_scores} {topk_inds}')

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _topk_with_center(scores, K=40, delta=1.0, multipiler=1.0):
    '''
    This function is similar to `_topk`, expect the scroes are weighted by the inverse-distance from the frame center. As the pure inserse function goes down too fast, I add two hyper-parameters here: `delta` and `multiplier`. It's not clear how to tune these two parameters.
    '''
    batch, cat, height, width = scores.size()
    # print('scores size: ', scores.size())

    weight_to_center = torch.zeros((height, width))
    y, x = np.ogrid[0:height, 0:width] # mli: borrowed from gaussian2D
    center_y, center_x = (height - 1) / 2.0, (width - 1)/ 2.0
    y = y - center_y
    x = x - center_x
    weight_to_center = multipiler / (np.sqrt(x * x + y * y) + delta)
    weight_to_center = torch.from_numpy(weight_to_center).to(scores.device)

    weight_to_center = weight_to_center.reshape(1, 1, height, width)
    scores *= weight_to_center

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs



def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
        wh = wh.view(batch, K, cat, 2)
        clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
        wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections


def multi_pose_decode(
        heat, wh, kps, reg=None, hm_hp=None, hp_offset=None, K=100):
    batch, cat, height, width = heat.size()
    num_joints = kps.shape[1] // 2
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    print(f'multi_pose_decode heat {inds}')

    kps = _transpose_and_gather_feat(kps, inds)
    print(f'multi_pose_decode kps  {kps.shape} {kps}')
    kps = kps.view(batch, K, num_joints * 2)
    kps[..., 1::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
    kps[..., 0::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)
    print(f'multi_pose_decode kps 1 {kps.shape} {kps}')
    # 转换为Y，X
    # kps = torch.stack([kps[..., 1::2], kps[..., 0::2]], dim=-1).view(batch, K, num_joints * 2)

    # kps[...,::2]
    if reg is not None:
        print(f'multi_pose_decode reg {reg.shape} inds {inds.shape}')
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1].clamp_(0,1)
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2].clamp_(0,1)
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    print(f'multi_pose_decode xs {xs.shape}')
    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    if hm_hp is not None:
        print(f'multi_pose_decode hm_hp {hm_hp.shape}')

        plt.figure()
        for i in range(17):
            plt.subplot(6, 3, i + 1)
            plt.title(joints_name[i])
            plt.imshow(hm_hp[0][i])
        plt.show()
        hm_hp = _nms(hm_hp)
        # import matplotlib.pyplot as plt
        # # plt.figure()
        # for i in range(15,17):
        #     plt.subplot(1, 3, i - 14)
        #     plt.imshow(hm_hp[0][i])
        # plt.show()
        # a = hm_hp.sum(dim = 1)
        # plt.imshow(a[0])
        # plt.show()
        thresh = 0.1
        kps = kps.view(batch, K, num_joints, 2).permute(
            0, 2, 1, 3).contiguous()  # b x J x K x 2
        reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
        # TODO:
        # 这里对于单人的情况必有问题 直接筛选最大值 可能导致连接多人
        # 应参考单人版本
        hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(
            hm_hp, K=K)  # b x J x K

        if hp_offset is not None:
            print(f'multi_pose_decode hp_offset {hp_offset.shape} hm_xs {hm_xs} hm_ys {hm_ys}')
            for i in range(34):
                plt.subplot(17,2, i + 1)
                plt.imshow(hp_offset[0,i])
                plt.title(joints_name[i//2])
            plt.show()
            # multi_pose_decode hp_offset torch.Size([1, 34, 64, 64])  hm_inds torch.Size([1, 17, 1])
            # hp_offset = _transpose_and_gather_feat(
            #     hp_offset, hm_inds.view(batch, -1))

            hp_offset = _transpose_and_gather_feat_plus(
                    hp_offset, hm_inds.view(batch, -1))
            print(f'multi_pose_decode hp_offset {hm_xs.shape} {hp_offset.shape} {hp_offset}')
            # multi_pose_decode hp_offset torch.Size([1, 17, 2])
            hp_offset = hp_offset.view(batch, num_joints, K, 2)
            # TODO: 检查 X Y
            # offet 最大为 1
            #
            hm_xs = hm_xs + hp_offset[:, :, :, 1].clamp_(0,1)
            hm_ys = hm_ys + hp_offset[:, :, :, 0].clamp_(0,1)
        else:
            hm_xs = hm_xs + 0.5
            hm_ys = hm_ys + 0.5

        print(f'multi_pose_decode hm_xs {hm_xs.shape}')

        # mask = (hm_score > thresh).float()
        # hm_score = (1 - mask) * -1 + mask * hm_score
        # hm_ys = (1 - mask) * (-10000) + mask * hm_ys
        # hm_xs = (1 - mask) * (-10000) + mask * hm_xs
        hm_kps = torch.stack([hm_ys, hm_xs], dim=-1).unsqueeze(
            2).expand(batch, num_joints, K, K, 2)
        print(f'multi_pose_decode hm_kps {hm_kps.shape} reg_kps {reg_kps.shape}')
        dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
        min_dist, min_ind = dist.min(dim=3)  # b x J x K
        hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
        min_dist = min_dist.unsqueeze(-1)
        min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
            batch, num_joints, K, 1, 2)
        hm_kps = hm_kps.gather(3, min_ind)
        hm_kps = hm_kps.view(batch, num_joints, K, 2)
        # l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(
        #     batch, num_joints, K, 1)
        # t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(
        #     batch, num_joints, K, 1)
        # r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(
        #     batch, num_joints, K, 1)
        # b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(
        #     batch, num_joints, K, 1)
        # mask = (hm_kps[..., 1:2] < l) + (hm_kps[..., 1:2] > r) + \
        #        (hm_kps[..., 0:1] < t) + (hm_kps[..., 0:1] > b) + \
        #        (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
        # mask = (mask > 0).float().expand(batch, num_joints, K, 2)
        # kps = (1 - mask) * hm_kps + mask * kps
        kps = hm_kps.permute(0, 2, 1, 3).contiguous().view(
            batch, K, num_joints * 2)
    detections = torch.cat([bboxes/height, scores, kps/height, clses], dim=2)

    return detections


def single_pose_decode(
        heat, wh, kps, reg=None, hm_hp=None, hp_offset=None, K=1):
    '''
      This function tries to reproduce the post-processing of MoveNet (https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html)
      Args:
        `heat`: the heatmap of human center point, with shape of (batch, 1, height, width)
        `wh`: the regression of width, height offsets from center point, with shape of (batch, 2, height, width)
        `kps`: the joint offsets from the center point, with shape of (batch, 17*2, height, width).
        `reg`: the center point offset, with shape of (batch, 2, height, width).
        `hm_hp`: the heatmap of joints, with shape of (batch, 17, height, width).
        `hp_offset`: the joint offsets from the regressed joint points, with shape of (batch, 2, height, width).
    '''
    batch, cat, height, width = heat.size()
    num_joints = kps.shape[1] // 2
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk_with_center(heat, K=K)
    kps = _transpose_and_gather_feat(kps, inds)
    # print('kps size: ', kps.size())
    kps = kps.view(batch, K, num_joints * 2)
    kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
    kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)


    xs = xs.view(batch, K, 1) + 0.5
    ys = ys.view(batch, K, 1) + 0.5
    # wh = _transpose_and_gather_feat(wh, inds)
    # wh = wh.view(batch, K, 2)
    # print('wh size: ', wh.size())
    # mli: produce dummy wh
    wh = torch.zeros((batch, K, 2)).to(xs.device)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    if hm_hp is not None:
        # print('hm_hp size: ', hm_hp.size())
        hm_hp = _nms(hm_hp)
        # thresh = 0.1
        thresh = 0.0 # mli: ignore the threshold here.
        kps = kps.view(batch, K, num_joints, 2).permute(
            0, 2, 1, 3).contiguous()  # b x J x K x 2
        reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
        # hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(
        #     hm_hp, K=K)  # b x J x K
        hm_score, hm_inds, hm_ys, hm_xs = _topk_channel_with_reg_kps(hm_hp, reg_kps, K=K) # b x J x K

        hp_offset = _transpose_and_gather_feat_plus(
            hp_offset, hm_inds.view(batch, -1))
        hp_offset = hp_offset.view(batch, num_joints, K, 2)
        hm_xs = hm_xs + hp_offset[:, :, :, 0]
        hm_ys = hm_ys + hp_offset[:, :, :, 1]


        mask = (hm_score > thresh).float()
        hm_score = (1 - mask) * -1 + mask * hm_score
        hm_ys = (1 - mask) * (-10000) + mask * hm_ys
        hm_xs = (1 - mask) * (-10000) + mask * hm_xs
        hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
            2).expand(batch, num_joints, K, K, 2)
        # print('hm_kps size: ', hm_kps.size())
        dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
        min_dist, min_ind = dist.min(dim=3)  # b x J x K
        hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
        min_dist = min_dist.unsqueeze(-1)
        min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
            batch, num_joints, K, 1, 2)
        hm_kps = hm_kps.gather(3, min_ind)
        hm_kps = hm_kps.view(batch, num_joints, K, 2)
        l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(
            batch, num_joints, K, 1)
        t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(
            batch, num_joints, K, 1)
        r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(
            batch, num_joints, K, 1)
        b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(
            batch, num_joints, K, 1)
        mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
               (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
               (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
        # print('mask size: ', mask.size())
        mask = (mask > 0).float().expand(batch, num_joints, K, 2)
        #   kps = (1 - mask) * hm_kps + mask * kps
        kps = hm_kps
        kps = kps.permute(0, 2, 1, 3).contiguous().view(
            batch, K, num_joints * 2)
    detections = torch.cat([bboxes, scores, kps, clses], dim=2)

    return detections
