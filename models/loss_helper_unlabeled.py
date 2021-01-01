""" Loss functions on unlabeled data

Written by Yezhen Cong, 2020
"""


import numpy as np
import torch
import torch.nn as nn

from models.ap_helper import flip_axis_to_camera
from models.loss_helper_iou import compute_iou_labels
from utils.box_util import get_3d_box
from utils.nms import lhs_3d_faster_samecls
from utils.nn_distance import nn_distance_withcls, nn_distance, huber_loss

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3  # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8]  # put larger weights on positive objectness
MAX_NUM_OBJ = 64


def trans_center(center, flip_x_axis, flip_y_axis, rot_mat, scale_ratio):
    """ teacher model bbox center -> student model bbox center
    """
    center_clone = center.clone()
    inds_to_flip_x_axis = torch.nonzero(flip_x_axis).squeeze(1)
    center_clone[inds_to_flip_x_axis, :, 0] = -center[inds_to_flip_x_axis, :, 0]

    inds_to_flip_y_axis = torch.nonzero(flip_y_axis).squeeze(1)
    center_clone[inds_to_flip_y_axis, :, 1] = -center[inds_to_flip_y_axis, :, 1]

    center_clone = torch.bmm(center_clone, rot_mat.transpose(1, 2))  # (B, num_proposal, 3)
    center_clone = center_clone * scale_ratio  # (B, K, 3) * (B, 1, 3)
    return center_clone


def trans_size(size_class, size_residual, scale_ratio, config):
    """ teacher model bbox size -> student model bbox size
    """
    mean_size_arr = config.mean_size_arr
    batch_size, num_proposal = size_class.shape[:2]
    mean_size_arr = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda()
    size_base = torch.index_select(mean_size_arr, 0, size_class.view(-1))
    size_base = size_base.view(batch_size, num_proposal, 3)
    size_all = size_base + size_residual
    size_all = size_all * scale_ratio
    size_residual_new = size_all - size_base
    return size_residual_new


# TODO check correctness
def trans_angle(angle_class, angle_residual, flip_x_axis, flip_y_axis, rot_angle, config):
    """ teacher model bbox heading -> student model bbox heading
        """
    angle = config.class2angle_gpu(angle_class, angle_residual)
    inds_to_flip_x_axis = torch.nonzero(flip_x_axis).squeeze(1)
    angle[inds_to_flip_x_axis, :] = np.pi - angle[inds_to_flip_x_axis, :]
    inds_to_flip_y_axis = torch.nonzero(flip_y_axis).squeeze(1)
    angle[inds_to_flip_y_axis, :] = -angle[inds_to_flip_y_axis, :]
    angle = angle - rot_angle.unsqueeze(-1)
    new_angle_class, new_angle_residual = config.angle2class_gpu(angle)
    return new_angle_class.long(), new_angle_residual


def reverse_trans_center(center, flip_x_axis, flip_y_axis, rot_mat, scale_ratio):
    """ student model bbox center -> teacher model bbox center
    """
    inds_to_flip_x_axis = torch.nonzero(flip_x_axis).squeeze(1)
    center_clone = center.clone()
    center_clone[inds_to_flip_x_axis, :, 0] = -center[inds_to_flip_x_axis, :, 0]

    inds_to_flip_y_axis = torch.nonzero(flip_y_axis).squeeze(1)
    center_clone[inds_to_flip_y_axis, :, 1] = -center[inds_to_flip_y_axis, :, 1]

    center_clone = torch.bmm(center_clone, rot_mat)  # not transpose
    center_clone = center_clone * (1 / scale_ratio)
    return center_clone


def compute_objectness_gt(end_points, unsupervised_inds):
    """ Compute cheating objectness loss for the proposals with GT labels
    Args:
        end_points: dict (read-only)
        unsupervised_inds: (batch_size, num_proposal) Tensor with value 0 or 1

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_proposal) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_proposal) Tensor with value 0 or 1
        object_assignment: (batch_size, num_proposal) Tensor with long int
            within [0,num_gt_object-1]
    """

    # Associate proposal and GT objects by point-to-point distances
    aggregated_vote_xyz = end_points['aggregated_vote_xyz'][unsupervised_inds, ...]
    gt_center = end_points['center_label'][unsupervised_inds, ...][:, :, 0:3]

    batch_size = gt_center.shape[0]
    num_proposal = aggregated_vote_xyz.shape[1]

    # ---- set the center of not gt placeholders to be -1000 -----
    gt_mask = (1 - end_points['box_label_mask'][unsupervised_inds, ...]).unsqueeze(-1).expand(-1, -1, 3).bool()
    gt_center[gt_mask] = -1000
    end_points['center_label'][unsupervised_inds, ...] = gt_center
    # ----------------

    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center)  # dist1: B_l xK, dist2: B_l xK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = torch.sqrt(dist1 + 1e-6)
    objectness_label = torch.zeros((batch_size, num_proposal), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((batch_size, num_proposal)).cuda()
    objectness_label[euclidean_dist1 < NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1 < NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1 > FAR_THRESHOLD] = 1

    # Compute objectness loss
    objectness_scores = end_points['objectness_scores'][unsupervised_inds, ...]
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    objectness_loss = criterion(objectness_scores.transpose(2, 1), objectness_label)
    mask_sum = (torch.sum(objectness_mask) + 1e-6)
    objectness_loss = torch.sum(objectness_loss * objectness_mask) / mask_sum

    object_assignment = ind1  # (batch_size, num_proposal) with values in 0,1,...,K2-1

    obj_pred_val = torch.argmax(end_points['objectness_scores'][unsupervised_inds, ...], 2)  # B,K
    obj_acc = torch.sum((obj_pred_val == objectness_label.long()).float() * objectness_mask) / mask_sum
    end_points['true_unlabeled_obj_acc'] = obj_acc  # this is true obj_acc

    return objectness_loss, objectness_label, objectness_mask, object_assignment


def compute_objectness_loss(end_points, unsupervised_inds, config_dict):
    """ Compute objectness loss for the proposals.

    Args:
        end_points: dict (read-only)
        unsupervised_inds: (batch_size, num_proposal) Tensor with value 0 or 1

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """

    # Associate proposal and GT objects by point-to-point distances
    aggregated_vote_xyz = end_points['aggregated_vote_xyz'][unsupervised_inds, ...]
    gt_center = end_points['unlabeled_center_label'][:, :, 0:3]

    batch_size = gt_center.shape[0]  # B_l
    num_proposal = aggregated_vote_xyz.shape[1]

    # ---- mod -----
    gt_mask = (1 - end_points['unlabeled_box_label_mask']).unsqueeze(-1).expand(-1, -1, 3).bool()
    gt_center[gt_mask] = -1000
    # ----------------

    #####################
    # may lose 15%-20% obj=1 (estimate)
    if config_dict['samecls_match']:
        dist1, ind1, dist2, ind2 = nn_distance_withcls(aggregated_vote_xyz, gt_center,
                                                       torch.argmax(
                                                           end_points['sem_cls_scores'][unsupervised_inds, ...], dim=2),
                                                       end_points['unlabeled_sem_cls_label'])
    else:
        dist1, ind1, dist2, ind2 = nn_distance(aggregated_vote_xyz, gt_center)  # dist1: B_l xK, dist2: B_l xK2

    ######################

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = torch.sqrt(dist1 + 1e-6)
    objectness_label = torch.zeros((batch_size, num_proposal), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((batch_size, num_proposal)).cuda()
    objectness_label[euclidean_dist1 < NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1 < NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1 > FAR_THRESHOLD] = 1

    # Compute objectness loss
    objectness_scores = end_points['objectness_scores'][unsupervised_inds, ...]
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    objectness_loss = criterion(objectness_scores.transpose(2, 1), objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask) / (torch.sum(objectness_mask) + 1e-6)

    # Set assignment
    object_assignment = ind1  # (batch_size, num_proposal) with values in 0,1,...,K2-1

    # only use these for cheating experiments
    return objectness_loss, objectness_label, objectness_mask, object_assignment


def compute_box_and_sem_cls_loss(end_points, unsupervised_inds, config, config_dict):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        end_points: dict (read-only)
        unsupervised_inds: (batch_size, num_proposal) Tensor with value 0 or 1

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    mean_size_arr = config.mean_size_arr

    object_assignment = end_points['unlabeled_object_assignment']
    box_label_mask = end_points['unlabeled_box_label_mask']
    objectness_label = end_points['unlabeled_objectness_label'].float()
    batch_size = object_assignment.shape[0]
    # Compute center loss
    dist1, ind1, dist2, _ = nn_distance(end_points['center'][unsupervised_inds, ...],
                                        end_points['unlabeled_center_label'][:, :, 0:3])  # dist1: BxK, dist2: BxK2

    centroid_reg_loss1 = \
        torch.sum(dist1 * objectness_label) / (torch.sum(objectness_label) + 1e-6)
    centroid_reg_loss2 = \
        torch.sum(dist2 * box_label_mask) / (torch.sum(box_label_mask) + 1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # Compute heading loss
    heading_class_label = torch.gather(end_points['unlabeled_heading_class_label'], 1,
                                       object_assignment)  # select (B,K) from (B,K2)
    criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
    heading_class_loss = criterion_heading_class(end_points['heading_scores'][unsupervised_inds, ...].transpose(2, 1),
                                                 heading_class_label)  # (B,K)
    heading_class_loss = torch.sum(heading_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

    heading_residual_label = torch.gather(end_points['unlabeled_heading_residual_label'], 1,
                                          object_assignment)  # select (B,K) from (B,K2)
    heading_residual_normalized_label = heading_residual_label / (np.pi / num_heading_bin)

    # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
    heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], num_heading_bin).zero_()
    heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1),
                                   1)  # src==1 so it's *one-hot* (B,K,num_heading_bin)
    heading_residual_pred = torch.sum(
        end_points['heading_residuals_normalized'][unsupervised_inds, ...] * heading_label_one_hot, -1)
    heading_diff = heading_residual_pred - heading_residual_normalized_label
    heading_residual_normalized_loss = huber_loss(heading_diff, delta=1.0)  # (B,K)
    heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss * objectness_label) / (
            torch.sum(objectness_label) + 1e-6)
    # Compute size loss
    size_class_label = torch.gather(end_points['unlabeled_size_class_label'], 1,
                                    object_assignment)  # select (B,K) from (B,K2)
    criterion_size_class = nn.CrossEntropyLoss(reduction='none')
    size_class_loss = criterion_size_class(end_points['size_scores'][unsupervised_inds, ...].transpose(2, 1),
                                           size_class_label)  # (B,K)
    size_class_loss = torch.sum(size_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

    size_residual_label = torch.gather(end_points['unlabeled_size_residual_label'], 1,
                                       object_assignment.unsqueeze(-1).repeat(1, 1, 3))  # select (B,K,3) from (B,K2,3)
    size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
    size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1)  # src==1 so it's *one-hot* (B,K,num_size_cluster)
    size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1, 1, 1, 3)  # (B,K,num_size_cluster,3)
    predicted_size_residual_normalized = torch.sum(
        end_points['size_residuals_normalized'][unsupervised_inds, ...] * size_label_one_hot_tiled,
        2)  # (B,K,3)

    mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(
        0)  # (1,1,num_size_cluster,3)
    mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2)  # (B,K,3)
    size_residual_label_normalized = size_residual_label / mean_size_label  # (B,K,3)
    size_diff = predicted_size_residual_normalized - size_residual_label_normalized
    size_residual_normalized_loss = torch.mean(huber_loss(size_diff, delta=1.0), -1)  # (B,K,3) -> (B,K)
    size_residual_normalized_loss = torch.sum(size_residual_normalized_loss * objectness_label) / (
            torch.sum(objectness_label) + 1e-6)

    # 3.4 Semantic cls loss
    sem_cls_label = torch.gather(end_points['unlabeled_sem_cls_label'], 1,
                                 object_assignment)  # select (B,K) from (B,K2)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    sem_cls_loss = criterion_sem_cls(end_points['sem_cls_scores'][unsupervised_inds, ...].transpose(2, 1),
                                     sem_cls_label)  # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

    return center_loss, heading_class_loss, heading_residual_normalized_loss, size_class_loss, size_residual_normalized_loss, sem_cls_loss


def get_pseudo_detection_loss(end_points, ema_end_points, config, config_dict):
    """ Loss functions for supervised samples in training detector

    Args:
        end_points: dict
            {
                seed_xyz, seed_inds, vote_xyz,
                center,
                heading_scores, heading_residuals_normalized,
                size_scores, size_residuals_normalized,
                sem_cls_scores, #seed_logits,#
                center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label,
                sem_cls_label,
                box_label_mask,
                vote_label, vote_label_mask
            }
        config: dataset config instance
    Returns:
        votenet_loss: pytorch scalar tensor
        end_points: dict
    """
    supervised_mask = end_points['supervised_mask']
    unsupervised_inds = torch.nonzero(1 - supervised_mask).squeeze(1).long()

    pseudo_objectness_loss, pseudo_objectness_label, pseudo_objectness_mask, pseudo_object_assignment = \
        compute_objectness_loss(end_points, unsupervised_inds, config_dict)

    if config_dict['view_stats']:
        gt_objectness_loss, gt_objectness_label, gt_objectness_mask, gt_object_assignment = \
            compute_objectness_gt(end_points, unsupervised_inds)  # calculate ground truth objectness

    end_points['unlabeled_objectness_loss'] = pseudo_objectness_loss
    end_points['unlabeled_objectness_label'] = pseudo_objectness_label
    end_points['unlabeled_objectness_mask'] = pseudo_objectness_mask
    end_points['unlabeled_object_assignment'] = pseudo_object_assignment
    total_num_proposal = pseudo_objectness_label.shape[0] * pseudo_objectness_label.shape[1]
    end_points['unlabeled_pos_ratio'] = \
        torch.sum(pseudo_objectness_label.float().cuda()) / float(total_num_proposal)
    end_points['unlabeled_neg_ratio'] = \
        torch.sum(pseudo_objectness_mask.float()) / float(total_num_proposal) - end_points['unlabeled_pos_ratio']
    # Box loss and sem cls loss
    center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = \
        compute_box_and_sem_cls_loss(end_points, unsupervised_inds, config, config_dict)

    end_points['unlabeled_center_loss'] = center_loss
    end_points['unlabeled_heading_cls_loss'] = heading_cls_loss
    end_points['unlabeled_heading_reg_loss'] = heading_reg_loss
    end_points['unlabeled_size_cls_loss'] = size_cls_loss
    end_points['unlabeled_size_reg_loss'] = size_reg_loss
    end_points['unlabeled_sem_cls_loss'] = sem_cls_loss
    box_loss = 0.1 * heading_cls_loss + heading_reg_loss + 0.1 * size_cls_loss + size_reg_loss + center_loss
    end_points['unlabeled_box_loss'] = box_loss

    unlabeled_loss = box_loss + 0.1 * sem_cls_loss

    unlabeled_loss *= 10
    end_points['unlabeled_detection_loss'] = unlabeled_loss

    # --------------------------------------------
    # Some other statistics
    if config_dict['view_stats']:
        obj_scores = end_points['objectness_scores'][unsupervised_inds, ...]
        obj_pred_val = torch.argmax(obj_scores, 2)  # B,K
        obj_acc = torch.sum((obj_pred_val == gt_objectness_label.long()).float() * gt_objectness_mask) / (
                torch.sum(gt_objectness_mask) + 1e-6)
        end_points['unlabeled_obj_acc'] = obj_acc

    return unlabeled_loss, end_points


def get_pseudo_labels(end_points, ema_end_points, pred_center, pred_sem_cls, pred_objectness, pred_heading_scores,
                      pred_heading_residuals,
                      pred_size_scores, pred_size_residuals, pred_vote_xyz, config_dict):
    batch_size, num_proposal = pred_center.shape[:2]
    label_mask = torch.zeros((batch_size, MAX_NUM_OBJ), dtype=torch.long).cuda()

    # obj score threshold
    pred_objectness = nn.Softmax(dim=2)(pred_objectness)
    # the second element is positive score
    pos_obj = pred_objectness[:, :, 1]
    neg_obj = pred_objectness[:, :, 0]
    objectness_mask = pos_obj > config_dict['obj_threshold']
    neg_objectness_mask = neg_obj > 0.9  # deprecated

    # cls score threshold
    pred_sem_cls = nn.Softmax(dim=2)(pred_sem_cls)
    max_cls, argmax_cls = torch.max(pred_sem_cls, dim=2)
    cls_mask = max_cls > config_dict['cls_threshold']

    supervised_mask = end_points['supervised_mask']
    unsupervised_inds = torch.nonzero(1 - supervised_mask).squeeze(1).long()

    iou_pred = nn.Sigmoid()(ema_end_points['iou_scores'][unsupervised_inds, ...])
    if iou_pred.shape[2] > 1:
        iou_pred = torch.gather(iou_pred, 2, argmax_cls.unsqueeze(-1)).squeeze(-1)  # use pred semantic labels
    else:
        iou_pred = iou_pred.squeeze(-1)

    if config_dict['view_stats']:
        # GT IoU labels (cheating) only for analyzing performance
        iou_labels, objectness_label, object_assignment = compute_iou_labels(
            end_points, unsupervised_inds, pred_vote_xyz,
            pred_center, pred_sem_cls, pred_objectness, pred_heading_scores,
            pred_heading_residuals, pred_size_scores, pred_size_residuals,
            config_dict)
        end_points['unlabeled_iou_labels'] = iou_labels
        end_points['unlabeled_pred_iou_value'] = torch.sum(iou_labels) / iou_labels.view(-1).shape[0]
        end_points['unlabeled_pred_iou_obj_value'] = torch.sum(iou_labels * objectness_label) / (
                torch.sum(objectness_label) + 1e-6)

        iou_acc = torch.abs(iou_pred - iou_labels)
        end_points['unlabeled_iou_acc'] = torch.sum(iou_acc) / iou_acc.view(-1).shape[0]
        obj_true_num = (torch.sum(objectness_label) + 1e-6)
        end_points['unlabeled_iou_obj_acc'] = torch.sum(iou_acc * objectness_label) / obj_true_num

        # for coverage calculation, associates every gt with pseudo labels
        gt_to_pseudo_iou = compute_iou_labels(
            end_points, unsupervised_inds, pred_vote_xyz,
            pred_center, pred_sem_cls, pred_objectness, pred_heading_scores,
            pred_heading_residuals, pred_size_scores, pred_size_residuals,
            config_dict, reverse=True)

    iou_threshold = config_dict['iou_threshold']
    iou_mask = iou_pred > iou_threshold
    before_iou_mask = torch.logical_and(cls_mask, objectness_mask)
    final_mask = torch.logical_and(before_iou_mask, iou_mask)

    # we only keep MAX_NUM_OBJ predictions
    # however, after filtering the number can still exceed this
    # so we keep the ones with larger pos_obj * max_cls
    inds = torch.argsort(pos_obj * max_cls * final_mask, dim=1, descending=True)

    inds = inds[:, :MAX_NUM_OBJ].long()
    final_mask_sorted = torch.gather(final_mask, dim=1, index=inds)
    end_points['pseudo_gt_ratio'] = torch.sum(final_mask_sorted).float() / final_mask_sorted.view(-1).shape[0]

    neg_objectness_mask = torch.gather(neg_objectness_mask, dim=1, index=inds)

    max_size, argmax_size = torch.max(pred_size_scores, dim=2)
    size_inds = argmax_size.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 3)
    max_heading, argmax_heading = torch.max(pred_heading_scores, dim=2)
    heading_inds = argmax_heading.unsqueeze(-1)

    # now only one class residuals
    pred_heading_residuals = torch.gather(pred_heading_residuals, dim=2, index=heading_inds).squeeze(2)
    pred_size_residuals = torch.gather(pred_size_residuals, dim=2, index=size_inds).squeeze(2)

    if config_dict['use_lhs']:
        pred_center_ = torch.gather(pred_center, dim=1, index=inds.unsqueeze(-1).expand(-1, -1, 3))
        pred_heading_class_ = torch.gather(argmax_heading, dim=1, index=inds)
        pred_heading_residual_ = torch.gather(pred_heading_residuals, dim=1, index=inds)
        pred_size_class_ = torch.gather(argmax_size, dim=1, index=inds)
        pred_size_residual_ = torch.gather(pred_size_residuals, dim=1, index=inds.unsqueeze(-1).expand(-1, -1, 3))
        num_proposal = pred_center_.shape[1]
        bsize = pred_center_.shape[0]
        pred_box_parameters = np.zeros((bsize, num_proposal, 7), dtype=np.float32)
        pred_box_parameters[:, :, 0:3] = pred_center_.detach().cpu().numpy()
        pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3), dtype=np.float32)
        pred_center_upright_camera = flip_axis_to_camera(pred_center_.detach().cpu().numpy())
        for i in range(bsize):
            for j in range(num_proposal):
                heading_angle = config_dict['dataset_config'].class2angle( \
                    pred_heading_class_[i, j].detach().cpu().numpy(),
                    pred_heading_residual_[i, j].detach().cpu().numpy())
                box_size = config_dict['dataset_config'].class2size( \
                    int(pred_size_class_[i, j].detach().cpu().numpy()),
                    pred_size_residual_[i, j].detach().cpu().numpy())
                pred_box_parameters[i, j, 3:6] = box_size
                pred_box_parameters[i, j, 6] = heading_angle
                corners_3d_upright_camera = get_3d_box(box_size, heading_angle, pred_center_upright_camera[i, j, :])
                pred_corners_3d_upright_camera[i, j] = corners_3d_upright_camera

        # pred_corners_3d_upright_camera, _ = predictions2corners3d(end_points, config_dict)
        pred_mask = np.ones((batch_size, MAX_NUM_OBJ))
        nonempty_box_mask = np.ones((batch_size, MAX_NUM_OBJ))
        pos_obj_numpy = torch.gather(pos_obj, dim=1, index=inds).detach().cpu().numpy()
        pred_sem_cls_numpy = torch.gather(argmax_cls, dim=1, index=inds).detach().cpu().numpy()
        iou_numpy = torch.gather(iou_pred, dim=1, index=inds).detach().cpu().numpy()
        for i in range(batch_size):
            boxes_3d_with_prob = np.zeros((MAX_NUM_OBJ, 8))
            for j in range(MAX_NUM_OBJ):
                boxes_3d_with_prob[j, 0] = np.min(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 1] = np.min(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 2] = np.min(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 3] = np.max(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 4] = np.max(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 5] = np.max(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 6] = pos_obj_numpy[i, j] * iou_numpy[i, j]
                boxes_3d_with_prob[j, 7] = pred_sem_cls_numpy[
                    i, j]  # only suppress if the two boxes are of the same class!!
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]

            # here we do not consider orientation, in accordance to test time nms
            pick = lhs_3d_faster_samecls(boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                                         config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert (len(pick) > 0)
            pred_mask[i, nonempty_box_inds[pick]] = 0
        # end_points['pred_mask'] = pred_mask
        final_mask_sorted[torch.from_numpy(pred_mask).bool().cuda()] = 0

    if config_dict['view_stats']:
        # ground truth coverage calculation
        selected_objectness_label = torch.gather(objectness_label, dim=1, index=inds)
        selected_object_assignment = torch.gather(object_assignment, dim=1, index=inds)
        gt_count = end_points['box_label_mask'].sum()

        picked_iou_labels = torch.gather(iou_labels, dim=1, index=inds)
        end_points['final_iou_avg_value'] = torch.sum(picked_iou_labels * final_mask_sorted).float() / (
                    torch.sum(final_mask_sorted) + 1e-6)
        end_points['final_iou_avg_obj_value'] = torch.sum(
            picked_iou_labels * final_mask_sorted * selected_objectness_label).float() / (
                                                        torch.sum(final_mask_sorted * selected_objectness_label) + 1e-6)
        selected_cls_pred = torch.gather(argmax_cls, dim=1, index=inds)
        selected_cls_gt = torch.gather(end_points['sem_cls_label'][unsupervised_inds, ...], dim=1,
                                       index=selected_object_assignment)
        correct_cls = selected_cls_pred == selected_cls_gt
        end_points['final_cls_value'] = torch.sum(
            correct_cls * final_mask_sorted).float() / (
                                                torch.sum(final_mask_sorted) + 1e-6)
        end_points['final_cls_obj_value'] = torch.sum(
            correct_cls * final_mask_sorted * selected_objectness_label).float() / (
                                                    torch.sum(final_mask_sorted * selected_objectness_label) + 1e-6)

        gt_to_pseudo_iou = torch.gather(gt_to_pseudo_iou, dim=2, index=inds.unsqueeze(1).expand(-1, 64, -1))
        gt_to_pseudo_iou = gt_to_pseudo_iou * final_mask_sorted.unsqueeze(1)
        gt_to_pseudo_iou = gt_to_pseudo_iou.max(dim=2)[0]
        range_25 = (gt_to_pseudo_iou > 0.25).float()
        range_5 = (gt_to_pseudo_iou > 0.5).float()
        end_points['final_coverage_0.25_value'] = torch.sum(range_25) / gt_count
        end_points['final_coverage_0.5_value'] = torch.sum(range_5) / gt_count

    label_mask[final_mask_sorted] = 1
    heading_label = torch.gather(argmax_heading, dim=1, index=inds)
    heading_residual_label = torch.gather(pred_heading_residuals.squeeze(-1), dim=1, index=inds)
    size_label = torch.gather(argmax_size, dim=1, index=inds)
    size_residual_label = torch.gather(pred_size_residuals, dim=1, index=inds.unsqueeze(-1).expand(-1, -1, 3))
    sem_cls_label = torch.gather(argmax_cls, dim=1, index=inds)
    center_label = torch.gather(pred_center, dim=1, index=inds.unsqueeze(-1).expand(-1, -1, 3))
    center_label[(1 - label_mask).unsqueeze(-1).expand(-1, -1, 3).bool()] = -1000
    false_center_label = torch.gather(pred_vote_xyz, dim=1, index=inds.unsqueeze(-1).expand(-1, -1, 3))
    false_center_label[torch.logical_not(neg_objectness_mask).unsqueeze(-1).expand(-1, -1, 3).bool()] = -1000

    iou_label = torch.gather(iou_pred, dim=1, index=inds)

    return label_mask, center_label, sem_cls_label, heading_label, heading_residual_label, size_label, size_residual_label, false_center_label, iou_label


def get_unlabeled_loss(end_points, ema_end_points, config, config_dict):
    unlabeled_batch_size = config_dict['unlabeled_batch_size']
    labeled_num = torch.nonzero(end_points['supervised_mask']).squeeze(1).shape[0]
    pred_center = ema_end_points['center'][labeled_num:]
    pred_sem_cls = ema_end_points['sem_cls_scores'][labeled_num:]
    pred_objectness = ema_end_points['objectness_scores'][labeled_num:]
    pred_heading_scores = ema_end_points['heading_scores'][labeled_num:]
    pred_heading_residuals = ema_end_points['heading_residuals'][labeled_num:]
    pred_size_scores = ema_end_points['size_scores'][labeled_num:]
    pred_size_residuals = ema_end_points['size_residuals'][labeled_num:]
    pred_vote_xyz = ema_end_points['aggregated_vote_xyz'][labeled_num:]

    # generate pseudo labels
    label_mask, center_label, sem_cls_label, \
    heading_label, heading_residual_label, \
    size_label, size_residual_label, \
    false_center_label, iou_label = \
        get_pseudo_labels(end_points, ema_end_points, pred_center, pred_sem_cls, pred_objectness, pred_heading_scores,
                          pred_heading_residuals, pred_size_scores, pred_size_residuals, pred_vote_xyz, config_dict)

    # center and size should be transformed
    center_label = trans_center(center_label, end_points['flip_x_axis'][labeled_num:],
                                end_points['flip_y_axis'][labeled_num:],
                                end_points['rot_mat'][labeled_num:], end_points['scale'][labeled_num:])
    false_center_label = trans_center(false_center_label, end_points['flip_x_axis'][labeled_num:],
                                      end_points['flip_y_axis'][labeled_num:],
                                      end_points['rot_mat'][labeled_num:], end_points['scale'][labeled_num:])
    size_residual_label = trans_size(size_label, size_residual_label, end_points['scale'][labeled_num:], config)
    if config_dict['dataset'] == 'sunrgbd':
        heading_label, heading_residual_label = trans_angle(heading_label, heading_residual_label,
                                                            end_points['flip_x_axis'][labeled_num:],
                                                            end_points['flip_y_axis'][labeled_num:],
                                                            end_points['rot_angle'][labeled_num:], config)

    if config_dict['view_stats']:
        # also transform gt labels for gt objectness cheating
        end_points['center_label'][labeled_num:] = trans_center(end_points['center_label'][labeled_num:],
                                                                end_points['flip_x_axis'][labeled_num:],
                                                                end_points['flip_y_axis'][labeled_num:],
                                                                end_points['rot_mat'][labeled_num:],
                                                                end_points['scale'][labeled_num:])

        end_points['size_residual_label'][labeled_num:] = trans_size(end_points['size_class_label'][labeled_num:],
                                                                     end_points['size_residual_label'][labeled_num:],
                                                                     end_points['scale'][labeled_num:],
                                                                     config)

    end_points['unlabeled_center_label'] = center_label
    end_points['unlabeled_box_label_mask'] = label_mask
    end_points['unlabeled_sem_cls_label'] = sem_cls_label
    end_points['unlabeled_heading_class_label'] = heading_label
    end_points['unlabeled_heading_residual_label'] = heading_residual_label
    end_points['unlabeled_size_class_label'] = size_label
    end_points['unlabeled_size_residual_label'] = size_residual_label
    end_points['unlabeled_false_center_label'] = false_center_label
    end_points['unlabeled_iou_label'] = iou_label

    consistency_loss, end_points = get_pseudo_detection_loss(end_points, ema_end_points, config, config_dict)

    return consistency_loss, end_points
