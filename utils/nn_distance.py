# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Chamfer distance in Pytorch.
Author: Charles R. Qi
Modified by: Yezhen Cong, 2020
"""

import torch
import torch.nn as nn
import numpy as np


def huber_loss(error, delta=1.0):
    """
    Args:
        error: Torch tensor (d1,d2,...,dk)
    Returns:
        loss: Torch tensor (d1,d2,...,dk)

    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    0.5 * d^2 + d * (|x|-d)     if |x|>d
    Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
    """
    abs_error = torch.abs(error)
    #quadratic = torch.min(abs_error, torch.FloatTensor([delta]))
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    loss = 0.5 * quadratic**2 + delta * linear
    return loss

def nn_distance(pc1, pc2, l1smooth=False, delta=1.0, l1=False):
    """
    Input:
        pc1: (B,N,C) torch tensor
        pc2: (B,M,C) torch tensor
        l1smooth: bool, whether to use l1smooth loss
        delta: scalar, the delta used in l1smooth loss
    Output:
        dist1: (B,N) torch float32 tensor
        idx1: (B,N) torch int64 tensor
        dist2: (B,M) torch float32 tensor
        idx2: (B,M) torch int64 tensor
    """
    N = pc1.shape[1]
    M = pc2.shape[1]
    pc1_expand_tile = pc1.unsqueeze(2).expand(-1,-1,M,-1)
    pc2_expand_tile = pc2.unsqueeze(1).expand(-1,N,-1,-1)
    pc_diff = pc1_expand_tile - pc2_expand_tile
    
    if l1smooth:
        pc_dist = torch.sum(huber_loss(pc_diff, delta), dim=-1) # (B,N,M)
    elif l1:
        pc_dist = torch.sum(torch.abs(pc_diff), dim=-1) # (B,N,M)
    else:
        pc_dist = torch.sum(pc_diff**2, dim=-1) # (B,N,M)
    dist1, idx1 = torch.min(pc_dist, dim=2) # (B,N)
    dist2, idx2 = torch.min(pc_dist, dim=1) # (B,M)
    return dist1, idx1, dist2, idx2


def nn_distance_exclude_self(pc1, pc2, l1smooth=False, delta=1.0, l1=False):
    """ Distance is calculated only if pc1 != pc2 (for nearest neighbour distance between a set of points and itself)
    Input:
        pc1: (B,N,C) torch tensor
        pc2: (B,M,C) torch tensor
        l1smooth: bool, whether to use l1smooth loss
        delta: scalar, the delta used in l1smooth loss
    Output:
        dist1: (B,N) torch float32 tensor
        idx1: (B,N) torch int64 tensor
        dist2: (B,M) torch float32 tensor
        idx2: (B,M) torch int64 tensor
    """
    N = pc1.shape[1]
    M = pc2.shape[1]
    B = pc1.shape[0]
    assert M == N
    pc1_expand_tile = pc1.unsqueeze(2).expand(-1, -1, M, -1)
    pc2_expand_tile = pc2.unsqueeze(1).expand(-1, N, -1, -1)

    diagonal_inds = torch.eye(N).cuda().unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, 3).bool()
    pc2_expand_tile = pc2_expand_tile.contiguous().view(B, -1, 3)
    pc2_expand_tile[diagonal_inds.view(B, -1, 3)] = -1000
    pc2_expand_tile = pc2_expand_tile.contiguous().view(B, -1, N, 3)
    pc_diff = pc1_expand_tile - pc2_expand_tile

    if l1smooth:
        pc_dist = torch.sum(huber_loss(pc_diff, delta), dim=-1)  # (B,N,M)
    elif l1:
        pc_dist = torch.sum(torch.abs(pc_diff), dim=-1)  # (B,N,M)
    else:
        pc_dist = torch.sum(pc_diff ** 2, dim=-1)  # (B,N,M)
    dist1, idx1 = torch.min(pc_dist, dim=2)  # (B,N)
    dist2, idx2 = torch.min(pc_dist, dim=1)  # (B,M)
    return dist1, idx1, dist2, idx2


def nn_distance_exclude_self_with_cls(pc1, pc2, cls1, cls2, l1smooth=False, delta=1.0, l1=False):
    """ Distance is calculated only if pc1 != pc2 (for nearest neighbour distance between a set of points and itself)
    Input:
        pc1: (B,N,C) torch tensor
        pc2: (B,M,C) torch tensor
        l1smooth: bool, whether to use l1smooth loss
        delta: scalar, the delta used in l1smooth loss
    Output:
        dist1: (B,N) torch float32 tensor
        idx1: (B,N) torch int64 tensor
        dist2: (B,M) torch float32 tensor
        idx2: (B,M) torch int64 tensor
    """
    N = pc1.shape[1]
    M = pc2.shape[1]
    B = pc1.shape[0]
    assert M == N
    pc1_expand_tile = pc1.unsqueeze(2).expand(-1, -1, M, -1)
    pc2_expand_tile = pc2.unsqueeze(1).expand(-1, N, -1, -1)

    diagonal_inds = torch.eye(N).cuda().unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, 3).bool()
    pc2_expand_tile = pc2_expand_tile.contiguous().view(B, -1, 3)
    pc2_expand_tile[diagonal_inds.view(B, -1, 3)] = -1000
    pc2_expand_tile = pc2_expand_tile.contiguous().view(B, -1, N, 3)
    pc_diff = pc1_expand_tile - pc2_expand_tile

    cls1_expand_tile = cls1.unsqueeze(2).expand(-1, -1, M)
    cls2_expand_tile = cls2.unsqueeze(1).expand(-1, N, -1)
    cls_mask = (cls1_expand_tile != cls2_expand_tile).int() * 1000

    if l1smooth:
        pc_dist = torch.sum(huber_loss(pc_diff, delta), dim=-1)  # (B,N,M)
    elif l1:
        pc_dist = torch.sum(torch.abs(pc_diff), dim=-1)  # (B,N,M)
    else:
        pc_dist = torch.sum(pc_diff ** 2, dim=-1)  # (B,N,M)
    pc_dist = pc_dist + cls_mask
    dist1, idx1 = torch.min(pc_dist, dim=2)  # (B,N)
    dist2, idx2 = torch.min(pc_dist, dim=1)  # (B,M)
    return dist1, idx1, dist2, idx2


def nn_distance_withcls(pc1, pc2, cls1, cls2, l1smooth=False, delta=1.0, l1=False):
    """ Distance is normally calculated only if cls1 == cls2
    Input:
        pc1: (B,N,C) torch tensor
        pc2: (B,M,C) torch tensor
        cls1: (B,N) torch tensor
        cls2: (B,M) torch tensor
        l1smooth: bool, whether to use l1smooth loss
        delta: scalar, the delta used in l1smooth loss
    Output:
        dist1: (B,N) torch float32 tensor
        idx1: (B,N) torch int64 tensor
        dist2: (B,M) torch float32 tensor
        idx2: (B,M) torch int64 tensor
    """
    N = pc1.shape[1]
    M = pc2.shape[1]
    pc1_expand_tile = pc1.unsqueeze(2).expand(-1, -1, M, -1)
    pc2_expand_tile = pc2.unsqueeze(1).expand(-1, N, -1, -1)
    pc_diff = pc1_expand_tile - pc2_expand_tile
    cls1_expand_tile = cls1.unsqueeze(2).expand(-1, -1, M)
    cls2_expand_tile = cls2.unsqueeze(1).expand(-1, N, -1)
    cls_mask = (cls1_expand_tile != cls2_expand_tile).int() * 1000

    if l1smooth:
        pc_dist = torch.sum(huber_loss(pc_diff, delta), dim=-1)  # (B,N,M)
    elif l1:
        pc_dist = torch.sum(torch.abs(pc_diff), dim=-1)  # (B,N,M)
    else:
        pc_dist = torch.sum(pc_diff ** 2, dim=-1)  # (B,N,M)

    pc_dist = pc_dist + cls_mask
    dist1, idx1 = torch.min(pc_dist, dim=2)  # (B,N)
    dist2, idx2 = torch.min(pc_dist, dim=1)  # (B,M)
    return dist1, idx1, dist2, idx2


def nn_distance_inbox(pc1, seed, pc2, half_size, l1smooth=False, delta=1.0, l1=False):
    """ This is for unsupervised vote loss calculation
    Input:
        pc1: (B,N,C) torch tensor
        pc2: (B,M,C) torch tensor
        half_size: (B,M,3) torch tensor
        l1smooth: bool, whether to use l1smooth loss
        delta: scalar, the delta used in l1smooth loss
    Output:
        dist1: (B,N) torch float32 tensor
        idx1: (B,N) torch int64 tensor
        dist2: (B,M) torch float32 tensor
        idx2: (B,M) torch int64 tensor
    """
    N = pc1.shape[1]
    M = pc2.shape[1]
    pc1_expand_tile = pc1.unsqueeze(2).expand(-1, -1, M, -1)
    seed_expand_tile = seed.unsqueeze(2).expand(-1, -1, M, -1)
    pc2_expand_tile = pc2.unsqueeze(1).expand(-1, N, -1, -1)
    pc_diff = pc1_expand_tile - pc2_expand_tile
    half_size_expand_tile = half_size.unsqueeze(1).expand(-1, N, -1, -1)
    lower = pc2_expand_tile - half_size_expand_tile
    higher = pc2_expand_tile + half_size_expand_tile
    in_box_mask = torch.logical_or((lower > seed_expand_tile).any(dim=3), (higher < seed_expand_tile).any(dim=3)).int() * 1000

    if l1smooth:
        pc_dist = torch.sum(huber_loss(pc_diff, delta), dim=-1)  # (B,N,M)
    elif l1:
        pc_dist = torch.sum(torch.abs(pc_diff), dim=-1)  # (B,N,M)
    else:
        pc_dist = torch.sum(pc_diff ** 2, dim=-1)  # (B,N,M)

    pc_dist = pc_dist + in_box_mask
    dist1, idx1 = torch.min(pc_dist, dim=2)  # (B,N)
    dist2, idx2 = torch.min(pc_dist, dim=1)  # (B,M)
    return dist1, idx1, dist2, idx2


def demo_nn_distance():
    np.random.seed(0)
    pc1arr = np.random.random((1,5,3))
    pc2arr = np.random.random((1,6,3))
    pc1 = torch.from_numpy(pc1arr.astype(np.float32))
    pc2 = torch.from_numpy(pc2arr.astype(np.float32))
    dist1, idx1, dist2, idx2 = nn_distance(pc1, pc2)
    print(dist1)
    print(idx1)
    dist = np.zeros((5,6))
    for i in range(5):
        for j in range(6):
            dist[i,j] = np.sum((pc1arr[0,i,:] - pc2arr[0,j,:]) ** 2)
    print(dist)
    print('-'*30)
    print('L1smooth dists:')
    dist1, idx1, dist2, idx2 = nn_distance(pc1, pc2, True)
    print(dist1)
    print(idx1)
    dist = np.zeros((5,6))
    for i in range(5):
        for j in range(6):
            error = np.abs(pc1arr[0,i,:] - pc2arr[0,j,:])
            quad = np.minimum(error, 1.0)
            linear = error - quad
            loss = 0.5*quad**2 + 1.0*linear
            dist[i,j] = np.sum(loss)
    print(dist)


if __name__ == '__main__':
    demo_nn_distance()
