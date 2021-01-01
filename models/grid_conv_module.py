""" Grid interpolation and convolution module for IoU estimation
Written by Yezhen Cong, 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

from utils.box_util import rot_gpu

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
from pointnet2.pointnet2_modules import PointnetSAModuleVotes
import pointnet2.pointnet2_utils as pointnet2_utils
import pointnet2.pytorch_utils as pt_utils


class GridConv(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling,
                 seed_feat_dim=256, query_feats='seed', iou_class_depend=True):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim
        self.query_feats = query_feats
        self.iou_class_depend = iou_class_depend

        # class dependent IoU
        self.iou_size = num_class if self.iou_class_depend else 1

        self.mlp_before_iou = pt_utils.SharedMLP([self.seed_feat_dim + 3, 128, 128, 128], bn=True)

        self.conv1_iou = torch.nn.Conv1d(128, 128, 1)
        self.conv2_iou = torch.nn.Conv1d(128, 128, 1)
        self.conv3_iou = torch.nn.Conv1d(128, 3 + num_heading_bin * 2 + num_size_cluster * 3 + self.iou_size, 1)
        self.bn1_iou = torch.nn.BatchNorm1d(128)
        self.bn2_iou = torch.nn.BatchNorm1d(128)

    def forward(self, center, size, heading, end_points):
        if self.query_feats == 'vote':
            origin_xyz = end_points['vote_xyz']
            origin_features = end_points['vote_features']
        elif self.query_feats == 'seed':
            origin_xyz = end_points['seed_xyz']
            origin_features = end_points['seed_features']
        elif self.query_feats == 'seed+vote':
            origin_xyz = end_points['seed_xyz']
            origin_features = end_points['vote_features']
        else:
            raise NotImplementedError()

        origin_features = origin_features.detach()
        origin_xyz = origin_xyz.detach()
        B, K = size.shape[:2]
        center_xyz = center
        grid_step = torch.linspace(-1, 1, 4).cuda()
        grid_size = 4
        grid_step_x = grid_step.view(grid_size, 1, 1).repeat(1, grid_size, grid_size)
        grid_step_x = grid_step_x.view(1, 1, -1).expand(B, K, -1)
        grid_step_y = grid_step.view(1, grid_size, 1).repeat(grid_size, 1, grid_size)
        grid_step_y = grid_step_y.view(1, 1, -1).expand(B, K, -1)
        grid_step_z = grid_step.view(1, 1, grid_size).repeat(grid_size, grid_size, 1)
        grid_step_z = grid_step_z.view(1, 1, -1).expand(B, K, -1)
        x_grid = grid_step_x * size[:, :, 0:1]
        y_grid = grid_step_y * size[:, :, 1:2]
        z_grid = grid_step_z * size[:, :, 2:3]
        whole_grid = torch.cat([x_grid.unsqueeze(-1), y_grid.unsqueeze(-1), z_grid.unsqueeze(-1)], dim=-1)

        rot_mat = rot_gpu(heading).view(-1, 3, 3)  # [B * S, 3, 3]
        whole_grid = torch.bmm(whole_grid.view(B * K, -1, 3), rot_mat.transpose(1, 2)).view(B, K, -1, 3)
        whole_grid = whole_grid + center_xyz.unsqueeze(2).expand(-1, -1, grid_size * grid_size * grid_size, -1)
        whole_grid = whole_grid.view(B, -1, 3).contiguous()

        origin_xyz = origin_xyz.contiguous()
        origin_features = origin_features.contiguous()

        feat_size = origin_features.shape[1]
        _, idx = pointnet2_utils.three_nn(whole_grid, origin_xyz)  # B, K*64, 3

        interp_points = torch.gather(origin_xyz, dim=1, index=idx.view(B, -1, 1).expand(-1, -1, 3).long())  # B, K*64*3, 3
        expanded_whole_grid = whole_grid.unsqueeze(2).expand(-1, -1, 3, -1).contiguous().view(B, -1, 3)  # B, K*64*3, 3
        dist = interp_points - expanded_whole_grid
        dist = torch.sqrt(torch.sum(dist * dist, dim=2))
        grid_point_num = grid_size * grid_size * grid_size
        relative_grid = whole_grid - center_xyz.unsqueeze(2).expand(-1, -1, grid_point_num, -1).contiguous().view(B, -1, 3)

        weight = 1 / (dist + 1e-8)
        weight = weight.view(B, -1, 3)
        norm = torch.sum(weight, dim=2, keepdim=True)
        weight = weight / norm
        weight = weight.contiguous()

        interpolated_feats = torch.cat([torch.index_select(a, 0, i).unsqueeze(0)
                                        for a, i in zip(origin_features.transpose(1, 2), idx.view(B, -1).long())], 0)
        interpolated_feats = torch.sum(interpolated_feats.view(B, -1, 3, feat_size) * weight.unsqueeze(-1), dim=2)
        interpolated_feats = interpolated_feats.transpose(1, 2)
        interpolated_feats = interpolated_feats.view(B, -1, K, grid_point_num)
        interpolated_feats = torch.cat([relative_grid.transpose(1, 2).contiguous().view(B, -1, K, 64), interpolated_feats], dim=1)
        interpolated_feats = self.mlp_before_iou(interpolated_feats) # B, C, K, 64

        iou_features = F.max_pool2d(interpolated_feats, kernel_size=[1, interpolated_feats.size(3)]).squeeze(-1)
        net_iou = F.relu(self.bn1_iou(self.conv1_iou(iou_features)))
        net_iou = F.relu(self.bn2_iou(self.conv2_iou(net_iou)))
        net_iou = self.conv3_iou(net_iou)

        end_points['iou_scores'] = net_iou.transpose(2, 1)[:, :, -self.iou_size:]
        return end_points
