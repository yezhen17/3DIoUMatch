""" IoU calculation functions

Written by Yezhen Cong, 2020
"""


import os
import sys
import torch

from utils.box_util import box3d_iou_batch_gpu, box3d_iou_gpu_axis_aligned

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from utils.nn_distance import nn_distance

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3


def compute_iou_from_given_size(end_points, unsupervised_inds, pred_center, pred_size, pred_heading, config_dict):
    center_label = end_points['center_label'][unsupervised_inds, ...]
    zero_mask = (1 - end_points['box_label_mask'][unsupervised_inds, ...]).unsqueeze(-1).expand(-1, -1, 3).bool()
    center_label[zero_mask] = -1000
    heading_class_label = end_points['heading_class_label'][unsupervised_inds, ...]
    heading_residual_label = end_points['heading_residual_label'][unsupervised_inds, ...]
    size_class_label = end_points['size_class_label'][unsupervised_inds, ...]
    size_residual_label = end_points['size_residual_label'][unsupervised_inds, ...]
    batch_size, num_proposal = pred_heading.shape[:2]

    # ------------------------- GT BBOX ----------------------------------------
    gt_size = config_dict['dataset_config'].class2size_gpu(size_class_label, size_residual_label)
    gt_angle = config_dict['dataset_config'].class2angle_gpu(heading_class_label, heading_residual_label)
    gt_bbox = torch.cat([center_label, gt_size, -gt_angle[:, :, None]], dim=2)
    pred_size[pred_size <= 0] = 1e-6
    pred_bbox = torch.cat([pred_center, pred_size, -pred_heading[:, :, None]], axis=2)

    end_points['pred_bbox'] = pred_bbox
    pred_num = pred_bbox.shape[1]
    gt_bbox_ = gt_bbox.view(-1, 7)
    pred_bbox_ = pred_bbox.view(-1, 7)
    iou_labels = box3d_iou_batch_gpu(pred_bbox_, gt_bbox_)
    iou_labels, object_assignment = iou_labels.view(batch_size * pred_num, batch_size, -1).max(dim=2)
    inds = torch.arange(batch_size).cuda().unsqueeze(1).expand(-1, pred_num).contiguous().view(-1, 1)
    iou_labels = iou_labels.gather(dim=1, index=inds).view(batch_size, -1)
    iou_labels = iou_labels.detach()
    object_assignment = object_assignment.gather(dim=1, index=inds).view(batch_size, -1)
    return iou_labels, None, object_assignment


def compute_iou_labels(end_points, unsupervised_inds, pred_votes, pred_center, pred_sem_cls, pred_objectness, pred_heading_scores,
                       pred_heading_residuals, pred_size_scores, pred_size_residuals, config_dict, reverse=False):

    # the end_points labels are not transformed
    center_label = end_points['center_label'][unsupervised_inds, ...]
    zero_mask = (1 - end_points['box_label_mask'][unsupervised_inds, ...]).unsqueeze(-1).expand(-1, -1, 3).bool()
    center_label[zero_mask] = -1000
    heading_class_label = end_points['heading_class_label'][unsupervised_inds, ...]
    heading_residual_label = end_points['heading_residual_label'][unsupervised_inds, ...]
    size_class_label = end_points['size_class_label'][unsupervised_inds, ...]
    size_residual_label = end_points['size_residual_label'][unsupervised_inds, ...]

    pred_heading_class = torch.argmax(pred_heading_scores, -1)
    pred_heading_residual = torch.gather(pred_heading_residuals, 2, pred_heading_class.unsqueeze(-1)).squeeze(2)
    pred_size_class = torch.argmax(pred_size_scores, -1)
    pred_size_class_inds = pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3)
    pred_size_residual = torch.gather(pred_size_residuals, 2, pred_size_class_inds).squeeze(2) # B, num_proposals, 3

    dist1, object_assignment, _, _ = nn_distance(pred_votes, center_label)
    euclidean_dist1 = torch.sqrt(dist1 + 1e-6)
    batch_size, num_proposal = euclidean_dist1.shape[:2]
    objectness_label = torch.zeros((batch_size, num_proposal), dtype=torch.long).cuda()
    objectness_label[euclidean_dist1 < NEAR_THRESHOLD] = 1

    # ------------------------- GT BBOX ----------------------------------------
    gt_size = config_dict['dataset_config'].class2size_gpu(size_class_label, size_residual_label)
    gt_angle = config_dict['dataset_config'].class2angle_gpu(heading_class_label, heading_residual_label)
    gt_bbox = torch.cat([center_label, gt_size, -gt_angle[:, :, None]], dim=2)

    pred_size = config_dict['dataset_config'].class2size_gpu(pred_size_class.detach(), pred_size_residual)
    pred_size[pred_size <= 0] = 1e-6

    if config_dict['dataset_config'].num_heading_bin == 1:
        pred_angle = torch.zeros(pred_size.shape[:2]).cuda()
    else:
        pred_angle = config_dict['dataset_config'].class2angle_gpu(pred_heading_class.detach(), pred_heading_residual)
    pred_bbox = torch.cat([pred_center, pred_size, -pred_angle[:, :, None]], axis=2)

    end_points['pred_bbox'] = pred_bbox
    pred_num = pred_bbox.shape[1]
    gt_num = gt_bbox.shape[1]

    # start = time.time()
    gt_bbox_ = gt_bbox.view(-1, 7)
    pred_bbox_ = pred_bbox.view(-1, 7)
    if reverse:
        iou_labels = box3d_iou_batch_gpu(gt_bbox_, pred_bbox_)
        iou_labels = iou_labels.view(batch_size * gt_num, batch_size, -1)
        inds = torch.arange(batch_size).cuda().unsqueeze(1).expand(-1, gt_num * pred_num).contiguous().view(-1, 1,
                                                                                                            pred_num)
        iou_labels = iou_labels.gather(dim=1, index=inds).view(batch_size, -1, pred_num)
        iou_labels = iou_labels.detach()
        return iou_labels
    else:
        iou_labels = box3d_iou_batch_gpu(pred_bbox_, gt_bbox_)
        iou_labels, object_assignment = iou_labels.view(batch_size * pred_num, batch_size, -1).max(dim=2)
        inds = torch.arange(batch_size).cuda().unsqueeze(1).expand(-1, pred_num).contiguous().view(-1, 1)
        iou_labels = iou_labels.gather(dim=1, index=inds).view(batch_size, -1)
        iou_labels = iou_labels.detach()
        object_assignment = object_assignment.gather(dim=1, index=inds).view(batch_size, -1)
        return iou_labels, objectness_label, object_assignment


def compute_iou_labels_axis_aligned_gpu(end_points, unsupervised_inds, pred_votes, pred_center, pred_sem_cls, pred_objectness, pred_heading_scores,
                       pred_heading_residuals, pred_size_scores, pred_size_residuals, config_dict):
    center_label = end_points['center_label'][unsupervised_inds, ...]
    zero_mask = (1 - end_points['box_label_mask'][unsupervised_inds, ...]).unsqueeze(-1).expand(-1, -1, 3).bool()
    center_label[zero_mask] = -1000
    size_class_label = end_points['size_class_label'][unsupervised_inds, ...]
    size_residual_label = end_points['size_residual_label'][unsupervised_inds, ...]
    origin_object_assignment = end_points['object_assignment'][unsupervised_inds, ...]

    dist1, object_assignment, _, _ = nn_distance(pred_votes, center_label)
    euclidean_dist1 = torch.sqrt(dist1 + 1e-6)
    batch_size, K = euclidean_dist1.shape[:2]
    objectness_label = torch.zeros((batch_size, K), dtype=torch.long).cuda()
    objectness_label[euclidean_dist1 < NEAR_THRESHOLD] = 1

    pred_size_class = torch.argmax(pred_size_scores, -1)  # B,num_proposal
    pred_size_residual = torch.gather(pred_size_residuals, 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3).detach())  # B,num_proposal,1,3
    pred_size_residual = pred_size_residual.squeeze(2)

    gt_size = config_dict['dataset_config'].class2size_gpu(size_class_label, size_residual_label) / 2
    gt_corners = torch.cat([(gt_size + center_label).unsqueeze(2), (center_label - gt_size).unsqueeze(2)], dim=2)
    pred_size = config_dict['dataset_config'].class2size_gpu(pred_size_class.detach(), pred_size_residual) / 2
    pred_corners = torch.cat([(pred_size + pred_center).unsqueeze(2), (pred_center - pred_size).unsqueeze(2)], dim=2)

    batch_size, pred_num = pred_corners.shape[:2]
    gt_num = gt_corners.shape[1]
    pred_corners_expand_tile = pred_corners.unsqueeze(2).expand(-1, -1, gt_num, -1, -1).contiguous().view(batch_size, -1, 2, 3)
    gt_corners_expand_tile = gt_corners.unsqueeze(1).expand(-1, pred_num, -1, -1, -1).contiguous().view(batch_size, -1, 2, 3)

    iou_labels = box3d_iou_gpu_axis_aligned(gt_corners_expand_tile.detach(), pred_corners_expand_tile)
    iou_labels, object_assignment = iou_labels.view(batch_size, pred_num, gt_num).max(2)

    iou_zero_mask = (iou_labels < 1e-4).int()
    final_object_assignment = origin_object_assignment * iou_zero_mask + object_assignment * (1 - iou_zero_mask)

    end_points['acc_pred_iou'] = torch.sum(iou_labels) / iou_labels.view(-1).shape[0]
    end_points['acc_pred_iou_obj'] = torch.sum(iou_labels * objectness_label) / (torch.sum(objectness_label) + 1e-6)
    return iou_labels, iou_zero_mask, final_object_assignment
