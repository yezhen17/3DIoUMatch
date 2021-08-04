# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Deep hough voting network for 3D object detection in point clouds.

Author: Charles R. Qi and Or Litany
"""

# Modified by Yezhen Cong, 2020

import numpy as np
import torch
import torch.nn as nn

from models.backbone_module import Pointnet2Backbone
from models.grid_conv_module import GridConv
from models.proposal_module import ProposalModule
from models.voting_module import VotingModule


class VoteNet(nn.Module):
    r"""
        A deep neural network for 3D object detection with end-to-end optimizable hough voting.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        vote_factor: (default: 1)
            Number of votes generated from each seed point.
    """

    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, dataset_config,
                 input_feature_dim=0, num_proposal=128, vote_factor=1, sampling='vote_fps',
                 query_feats='seed',
                 ):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.dataset_config = dataset_config
        assert (mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.dataset_config = dataset_config

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and detection
        self.pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster,
                                   mean_size_arr, num_proposal, sampling, query_feats=query_feats)

        self.grid_conv = GridConv(
            num_class, num_heading_bin, num_size_cluster,
            mean_size_arr, num_proposal, sampling, query_feats=query_feats,
        )

    def forward_backbone(self, inputs):
        """ Forward a pass through backbone but not iou branch

                Args:
                    inputs: dict
                        {point_clouds}

                        point_clouds: Variable(torch.cuda.FloatTensor)
                            (B, N, 3 + input_channels) tensor
                            Point cloud to run predicts on
                            Each point in the point-cloud MUST
                            be formatted as (x, y, z, features...)
                Returns:
                    end_points: dict
                """
        end_points = {}
        batch_size = inputs['point_clouds'].shape[0]
        end_points = self.backbone_net(inputs['point_clouds'], end_points)

        # --------- HOUGH VOTING ---------
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features

        xyz, features = self.vgen(xyz, features)

        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        end_points['vote_xyz'] = xyz
        end_points['vote_features'] = features

        end_points = self.pnet(xyz, features, end_points)
        return end_points

    def calculate_bbox(self, end_points):
        # calculate size and center
        size_scores = end_points['size_scores']
        size_residuals = end_points['size_residuals']
        B, K = size_scores.shape[:2]
        mean_size_arr = self.mean_size_arr
        mean_size_arr = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda()  # (num_size_cluster,3)
        size_class = torch.argmax(size_scores, -1)  # B,num_proposal
        size_residual = torch.gather(size_residuals, 2,
                                     size_class.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 3))  # B,num_proposal,1,3
        size_residual = size_residual.squeeze(2)
        size_base = torch.index_select(mean_size_arr, 0, size_class.view(-1))
        size_base = size_base.view(B, K, 3)
        size = (size_base + size_residual) / 2  # half of the size
        size[size < 0] = 1e-6
        center = end_points['center']

        heading_scores = end_points['heading_scores']
        heading_class = torch.argmax(heading_scores, -1)  # B,num_proposal
        heading_residuals = end_points['heading_residuals']
        heading_residual = torch.gather(heading_residuals, 2, heading_class.unsqueeze(-1))  # B,num_proposal,1
        heading_residual = heading_residual.squeeze(2)
        heading = self.dataset_config.class2angle_gpu(heading_class, heading_residual)

        end_points['size'] = size
        end_points['heading'] = heading
        return center, size, heading

    def forward(self, inputs, iou_opt=False):
        end_points = self.forward_backbone(inputs)
        center, size, heading = self.calculate_bbox(end_points)

        if iou_opt:
            center.retain_grad()
            size.retain_grad()
            if heading.requires_grad:
                heading.retain_grad()
            end_points = self.grid_conv(center, size, heading, end_points)
        else:
            end_points = self.grid_conv(center.detach(), size.detach(), heading.detach(), end_points)
        return end_points

    def forward_iou_part_only(self, end_points, center, size):
        end_points = self.grid_conv(center, size, end_points)
        return end_points

    def forward_with_pred_jitter(self, inputs):
        end_points = self.forward_backbone(inputs)
        center, size, heading = self.calculate_bbox(end_points)
        B, origin_proposal_num = heading.shape[0:2]

        factor = 1
        center_jitter = center.unsqueeze(2).expand(-1, -1, factor, -1).contiguous().view(B, -1, 3)
        size_jitter = size.unsqueeze(2).expand(-1, -1, factor, -1).contiguous().view(B, -1, 3)
        heading_jitter = heading.unsqueeze(2).expand(-1, -1, factor).contiguous().view(B, -1)
        center_jitter = center_jitter + size_jitter * torch.randn(size_jitter.shape).cuda() * 0.3
        size_jitter = size_jitter + size_jitter * torch.randn(size_jitter.shape).cuda() * 0.3
        size_jitter = torch.clamp(size_jitter, min=1e-8)

        center = torch.cat([center, center_jitter], dim=1)
        size = torch.cat([size, size_jitter], dim=1)
        heading = torch.cat([heading, heading_jitter], dim=1)

        end_points = self.grid_conv(center.detach(), size.detach(), heading.detach(), end_points)
        end_points['iou_scores_jitter'] = end_points['iou_scores'][:, origin_proposal_num:]
        end_points['iou_scores'] = end_points['iou_scores'][:, :origin_proposal_num]

        end_points['jitter_center'] = center_jitter
        end_points['jitter_size'] = size_jitter * 2
        end_points['jitter_heading'] = heading_jitter
        return end_points

    def forward_onlyiou_faster(self, end_points, center, size, heading):
        end_points = self.grid_conv(center, size, heading, end_points)
        return end_points



