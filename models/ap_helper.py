# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Modified by Zhao Na, 2019

# Modified by Yezhen Cong, 2020

""" Helper functions and class to calculate Average Precisions for 3D object detection.
"""
import os
import sys

import numpy as np
import torch
from torch import nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from utils.eval_det import eval_det_multiprocessing, get_iou_obb
from utils.nms import nms_2d_faster, nms_3d_faster, nms_3d_faster_samecls
from utils.box_util import get_3d_box, box3d_iou
from sunrgbd.sunrgbd_utils import extract_pc_in_box3d
from utils.pc_util import random_sampling

def flip_axis_to_camera(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    '''
    pc2 = np.copy(pc)
    pc2[...,[0,1,2]] = pc2[...,[0,2,1]] # cam X,Y,Z = depth X,-Z,Y
    pc2[...,1] *= -1
    return pc2

def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[...,[0,1,2]] = pc2[...,[0,2,1]] # depth X,Y,Z = cam X,Z,-Y
    pc2[...,2] *= -1
    return pc2

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs


def predictions2corners3d(end_points, config_dict):
    """ Convert predictions to OBB parameters (eight corner points)
    Args:
        end_points: dict
            {point_clouds, center, heading_scores, heading_residuals,
            size_scores, size_residuals, sem_cls_scores}
        config_dict: dict
            {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
            use_old_type_nms, conf_thresh, per_class_proposal}
    Returns:
        pred_corners_3d_upright_camera: ndarray (num_batch, num_proposals, 8, 3)
        pred_box_parameters:  ndarray (num_batch, num_proposals, 7)
    """
    pred_center = end_points['center'] # B,num_proposal,3
    pred_heading_class = torch.argmax(end_points['heading_scores'], -1) # B,num_proposal
    pred_heading_residual = torch.gather(end_points['heading_residuals'], 2,
        pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
    pred_heading_residual.squeeze_(2)
    pred_size_class = torch.argmax(end_points['size_scores'], -1) # B,num_proposal
    pred_size_residual = torch.gather(end_points['size_residuals'], 2,
        pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
    pred_size_residual.squeeze_(2)

    num_proposal = pred_center.shape[1]
    # Since we operate in upright_depth coord for points, while util functions
    # assume upright_camera coord.
    bsize = pred_center.shape[0]
    pred_box_parameters = np.zeros((bsize, num_proposal, 7), dtype=np.float32)
    pred_box_parameters[:,:,0:3] = pred_center.detach().cpu().numpy()
    pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3), dtype=np.float32)
    pred_center_upright_camera = flip_axis_to_camera(pred_center.detach().cpu().numpy())
    for i in range(bsize):
        for j in range(num_proposal):
            heading_angle = config_dict['dataset_config'].class2angle(\
                pred_heading_class[i,j].detach().cpu().numpy(), pred_heading_residual[i,j].detach().cpu().numpy())
            box_size = config_dict['dataset_config'].class2size(\
                int(pred_size_class[i,j].detach().cpu().numpy()), pred_size_residual[i,j].detach().cpu().numpy())
            pred_box_parameters[i,j,3:6] = box_size
            pred_box_parameters[i,j,6] = heading_angle
            corners_3d_upright_camera = get_3d_box(box_size, heading_angle, pred_center_upright_camera[i,j,:])
            pred_corners_3d_upright_camera[i,j] = corners_3d_upright_camera

    return pred_corners_3d_upright_camera, pred_box_parameters


def parse_predictions(end_points, config_dict):
    """ Parse predictions to OBB parameters and suppress overlapping boxes
    
    Args:
        end_points: dict
            {point_clouds, center, heading_scores, heading_residuals,
            size_scores, size_residuals, sem_cls_scores}
        config_dict: dict
            {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
            use_old_type_nms, conf_thresh, per_class_proposal}

    Returns:
        batch_pred_map_cls: a list of len == batch size (BS)
            [pred_list_i], i = 0, 1, ..., BS-1
            where pred_list_i = [(pred_sem_cls, box_params, box_score)_j]
            where j = 0, ..., num of valid detections - 1 from sample input i
    """
    pred_center = end_points['center']  # B,num_proposal,3
    pred_sem_cls = torch.argmax(end_points['sem_cls_scores'], -1) # B,num_proposal
    sem_cls_probs = softmax(end_points['sem_cls_scores'].detach().cpu().numpy()) # B,num_proposal,10
    pred_sem_cls_prob = np.max(sem_cls_probs,-1) # B,num_proposal

    pred_corners_3d_upright_camera, pred_box_parameters = predictions2corners3d(end_points, config_dict)
    bsize = pred_corners_3d_upright_camera.shape[0]
    K = pred_corners_3d_upright_camera.shape[1] # K==num_proposal
    nonempty_box_mask = np.ones((bsize, K))

    if config_dict['remove_empty_box']:
        # -------------------------------------
        # Remove predicted boxes without any point within them..
        batch_pc = end_points['point_clouds'].cpu().numpy()[:,:,0:3] # B,N,3
        for i in range(bsize):
            pc = batch_pc[i,:,:] # (N,3)
            for j in range(K):
                box3d = pred_corners_3d_upright_camera[i,j,:,:] # (8,3)
                box3d = flip_axis_to_depth(box3d)
                pc_in_box,inds = extract_pc_in_box3d(pc, box3d)
                if len(pc_in_box) < 5:
                    nonempty_box_mask[i,j] = 0
        # -------------------------------------

    obj_logits = end_points['objectness_scores'].detach().cpu().numpy()
    obj_prob = softmax(obj_logits)[:,:,1] # (B,K)
    if not config_dict['use_3d_nms']:
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_2d_with_prob = np.zeros((K,5))
            for j in range(K):
                boxes_2d_with_prob[j,0] = np.min(pred_corners_3d_upright_camera[i,j,:,0])
                boxes_2d_with_prob[j,2] = np.max(pred_corners_3d_upright_camera[i,j,:,0])
                boxes_2d_with_prob[j,1] = np.min(pred_corners_3d_upright_camera[i,j,:,2])
                boxes_2d_with_prob[j,3] = np.max(pred_corners_3d_upright_camera[i,j,:,2])
                boxes_2d_with_prob[j,4] = obj_prob[i,j]
            nonempty_box_inds = np.where(nonempty_box_mask[i,:]==1)[0]
            pick = nms_2d_faster(boxes_2d_with_prob[nonempty_box_mask[i,:] == 1, :],
                                 config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert(len(pick)>0)
            pred_mask[i, nonempty_box_inds[pick]] = 1
        end_points['pred_mask'] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict['use_3d_nms'] and (not config_dict['cls_nms']):
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K,7))
            for j in range(K):
                boxes_3d_with_prob[j,0] = np.min(pred_corners_3d_upright_camera[i,j,:,0])
                boxes_3d_with_prob[j,1] = np.min(pred_corners_3d_upright_camera[i,j,:,1])
                boxes_3d_with_prob[j,2] = np.min(pred_corners_3d_upright_camera[i,j,:,2])
                boxes_3d_with_prob[j,3] = np.max(pred_corners_3d_upright_camera[i,j,:,0])
                boxes_3d_with_prob[j,4] = np.max(pred_corners_3d_upright_camera[i,j,:,1])
                boxes_3d_with_prob[j,5] = np.max(pred_corners_3d_upright_camera[i,j,:,2])
                boxes_3d_with_prob[j,6] = obj_prob[i,j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_3d_faster(boxes_3d_with_prob[nonempty_box_mask[i,:] == 1, :],
                                 config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert(len(pick)>0)
            pred_mask[i, nonempty_box_inds[pick]] = 1
        end_points['pred_mask'] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict['use_3d_nms'] and config_dict['cls_nms']:
        # ---------- NMS input: pred_with_prob in (B,K,8) -----------
        scores = obj_prob
        if config_dict['use_iou_for_nms']:
            iou_logits = nn.Sigmoid()(end_points['iou_scores'])
            if iou_logits.shape[2] > 1:
                iou_logits = torch.gather(iou_logits, 2, pred_sem_cls.unsqueeze(-1))
            iou_logits = iou_logits.squeeze(-1).detach().cpu().numpy()
            scores = scores * iou_logits
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K,8))
            for j in range(K):
                boxes_3d_with_prob[j,0] = np.min(pred_corners_3d_upright_camera[i,j,:,0])
                boxes_3d_with_prob[j,1] = np.min(pred_corners_3d_upright_camera[i,j,:,1])
                boxes_3d_with_prob[j,2] = np.min(pred_corners_3d_upright_camera[i,j,:,2])
                boxes_3d_with_prob[j,3] = np.max(pred_corners_3d_upright_camera[i,j,:,0])
                boxes_3d_with_prob[j,4] = np.max(pred_corners_3d_upright_camera[i,j,:,1])
                boxes_3d_with_prob[j,5] = np.max(pred_corners_3d_upright_camera[i,j,:,2])
                boxes_3d_with_prob[j,6] = scores[i,j]
                boxes_3d_with_prob[j,7] = pred_sem_cls[i,j] # only suppress if the two boxes are of the same class!!
            nonempty_box_inds = np.where(nonempty_box_mask[i,:]==1)[0]
            pick = nms_3d_faster_samecls(boxes_3d_with_prob[nonempty_box_mask[i,:]==1,:],
                config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert(len(pick)>0)
            pred_mask[i, nonempty_box_inds[pick]] = 1
        end_points['pred_mask'] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------

    # a list (len: batch_size) of list (len: num of predictions per sample)
    # of tuples of pred_cls, pred_box and conf (0-1)
    batch_pred_map_cls = []
    for i in range(bsize):
        if config_dict['per_class_proposal']:
            cur_list = []
            for ii in range(config_dict['dataset_config'].num_class):
                cur_list += [(ii, pred_corners_3d_upright_camera[i,j], sem_cls_probs[i,j,ii]*obj_prob[i,j]) \
                    for j in range(pred_center.shape[1]) if pred_mask[i,j]==1 and obj_prob[i,j]>config_dict['conf_thresh']]
            batch_pred_map_cls.append(cur_list)
        else:
            batch_pred_map_cls.append([(pred_sem_cls[i,j].item(), pred_corners_3d_upright_camera[i,j], obj_prob[i,j]) \
                for j in range(pred_center.shape[1]) if pred_mask[i,j]==1 and obj_prob[i,j]>config_dict['conf_thresh']])
    end_points['batch_pred_map_cls'] = batch_pred_map_cls

    return batch_pred_map_cls


def groundtruths2corners3d(end_points, config_dict):
    """ Convert predictions to OBB parameters (eight corner points)
    Args:
        end_points: dict
            {center_label, heading_class_label, heading_residual_label,
            size_class_label, size_residual_label, sem_cls_label,
            box_label_mask}
        config_dict: dict
            {dataset_config}
    Returns:
        gt_corners_3d_upright_camera: ndarray (num_batch, MAX_NUM_OBJ, 8, 3)
        gt_box_parameters:  ndarray (num_batch, num_proposals, 7)
    """
    center_label = end_points['center_label']
    heading_class_label = end_points['heading_class_label']
    heading_residual_label = end_points['heading_residual_label']
    size_class_label = end_points['size_class_label']
    size_residual_label = end_points['size_residual_label']
    box_label_mask = end_points['box_label_mask']
    bsize = center_label.shape[0]

    K2 = center_label.shape[1] # K2==MAX_NUM_OBJ
    gt_box_parameters = np.zeros((bsize, K2, 7), dtype=np.float32)
    gt_box_parameters[:, :, 0:3] = center_label.detach().cpu().numpy()
    gt_corners_3d_upright_camera = np.zeros((bsize, K2, 8, 3), dtype=np.float32)
    gt_center_upright_camera = flip_axis_to_camera(center_label[:,:,0:3].detach().cpu().numpy())
    for i in range(bsize):
        for j in range(K2):
            if box_label_mask[i,j] == 0: continue
            heading_angle = config_dict['dataset_config'].class2angle(heading_class_label[i,j].detach().cpu().numpy(), heading_residual_label[i,j].detach().cpu().numpy())
            box_size = config_dict['dataset_config'].class2size(int(size_class_label[i,j].detach().cpu().numpy()), size_residual_label[i,j].detach().cpu().numpy())
            gt_box_parameters[i,j,3:6] = box_size
            gt_box_parameters[i,j,6] = heading_angle
            corners_3d_upright_camera = get_3d_box(box_size, heading_angle, gt_center_upright_camera[i,j,:])
            gt_corners_3d_upright_camera[i,j] = corners_3d_upright_camera

    return gt_corners_3d_upright_camera, gt_box_parameters

def parse_groundtruths(end_points, config_dict):
    """ Parse groundtruth labels to OBB parameters.
    
    Args:
        end_points: dict
            {center_label, heading_class_label, heading_residual_label,
            size_class_label, size_residual_label, sem_cls_label,
            box_label_mask}
        config_dict: dict
            {dataset_config}

    Returns:
        batch_gt_map_cls: a list  of len == batch_size (BS)
            [gt_list_i], i = 0, 1, ..., BS-1
            where gt_list_i = [(gt_sem_cls, gt_box_params)_j]
            where j = 0, ..., num of objects - 1 at sample input i
    """
    sem_cls_label = end_points['sem_cls_label']
    box_label_mask = end_points['box_label_mask']

    gt_corners_3d_upright_camera, _ = groundtruths2corners3d(end_points, config_dict)
    bsize = gt_corners_3d_upright_camera.shape[0]

    batch_gt_map_cls = []
    for i in range(bsize):
        batch_gt_map_cls.append([(sem_cls_label[i,j].item(), gt_corners_3d_upright_camera[i,j]) for j in range(gt_corners_3d_upright_camera.shape[1]) if box_label_mask[i,j]==1])
    end_points['batch_gt_map_cls'] = batch_gt_map_cls

    return batch_gt_map_cls


def align_predictions_groundtruths(batch_pred_corners_3d, batch_gt_corners_3d, end_points, iou_threshold=0.5):
    """

    Args:
        batch_pred_corners_3d: ndarray (num_batch, num_proposals, 8, 3)
            predicted bounding boxes (represented by 8 corner points) in upright_camera coordinate
        batch_gt_corners_3d: ndarray (num_batch, MAX_NUM_OBJ, 8, 3)
            ground truth bounding boxes (represented by 8 corner points) in upright_camera coordinate
        end_points: dict
            {box_label_mask, ...}
    Returns:
        batch_gt_corners_3d_aligned: ndarray (num_batch, num_proposals, 8, 3)
            clostest ground truth bounding boxes corresponding to each predicted bbox
        batch_confidence_scores: ndarray (num_batch, num_proposals, 1), value is 0 or 1
            the fitness between each predicted bbox and gt bbox, if the overlap larger than threshold, fitness is 1
        batch_sem_cls_labels: ndarray  (num_batch, num_proposals), value is [0, num_class-1]
            the semantic class of the aligned ground truth bboxes
    """
    bsize = batch_pred_corners_3d.shape[0]
    num_proposal = batch_pred_corners_3d.shape[1]
    box_label_mask = end_points['box_label_mask'].detach().cpu().numpy()
    sem_cls_label = end_points['sem_cls_label'].detach().cpu().numpy()

    batch_sem_cls_labels = np.zeros((bsize, num_proposal,1), dtype=np.int64)
    batch_confidence_scores = np.zeros((bsize, num_proposal,1), dtype=np.float32)
    batch_gt_corners_3d_aligned = np.zeros((bsize, num_proposal, 8, 3), dtype=np.float32)

    for i in range(bsize):
        cur_mask = np.nonzero(box_label_mask[i])
        gt_corners_3d = batch_gt_corners_3d[i][cur_mask]
        gt_classes = sem_cls_label[i][cur_mask]
        for j in range(num_proposal):
            BB = batch_pred_corners_3d[i,j,:,:]
            iou_list = []
            for BBGT in gt_corners_3d:
                iou, _ = box3d_iou(BB, BBGT)
                iou_list.append(iou)
            if len(iou_list) != 0:
                iou_list = np.array(iou_list)
                max_ind = np.argmax(iou_list)
                batch_gt_corners_3d_aligned[i,j,:,:] = gt_corners_3d[max_ind]
                batch_sem_cls_labels[i,j] = gt_classes[max_ind]
                if iou_list.max() >= iou_threshold:
                    batch_confidence_scores[i,j] = 1.

    return batch_gt_corners_3d_aligned, batch_confidence_scores, batch_sem_cls_labels


def get_roi_ptcloud(inputs, batch_pred_boxes_params, enlarge_ratio=1.2, num_point_roi=512, min_num_point=100):
    """ Generate ROI point cloud w.r.t predicted box

    :param inputs: dict {'point_clouds'}
                   input point clouds of the whole scene
           batch_pred_boxes_params: (B, num_proposals, 7), numpy array
                   predicted bounding box from detector
           enlarge_ratio: scalar
                   the value to enlarge the predicted box size
           num_point_roi: scalar
                   the number of points to be sampled in each enlarged box

    :return:
        batch_pc_roi: (B, num_proposals, num_sampled_points, input_pc_features) numpy array
        nonempty_roi_mask: (B, num_proposals) numpy array
    """
    batch_pc = inputs['point_clouds'].detach().cpu().numpy()[:, :, :]  # B,N,C
    bsize = batch_pred_boxes_params.shape[0]
    K = batch_pred_boxes_params.shape[1]
    batch_pc_roi = np.zeros((bsize, K, num_point_roi, batch_pc.shape[2]), dtype=np.float32)
    nonempty_roi_mask = np.ones((bsize, K))

    for i in range(bsize):
        pc = batch_pc[i, :, :]  # (N,C)
        for j in range(K):
            box_params = batch_pred_boxes_params[i, j, :]  # (7)
            center = box_params[0:3]
            center_upright_camera = flip_axis_to_camera(center)#.reshape(1,-1))[0]
            box_size = box_params[3:6]*enlarge_ratio #enlarge the box size
            heading_angle = box_params[6]
            box3d = get_3d_box(box_size, heading_angle, center_upright_camera)
            box3d = flip_axis_to_depth(box3d)
            pc_in_box, inds = extract_pc_in_box3d(pc, box3d)
            # print('The number of points in roi box is ', pc_in_box.shape[0])
            if len(pc_in_box) >= min_num_point:
                batch_pc_roi[i, j, :, :] = random_sampling(pc_in_box, num_point_roi)
            else:
                nonempty_roi_mask[i,j] = 0
    return batch_pc_roi, nonempty_roi_mask


class APCalculator(object):
    ''' Calculating Average Precision '''
    def __init__(self, ap_iou_thresh=0.25, class2type_map=None):
        """
        Args:
            ap_iou_thresh: float between 0 and 1.0
                IoU threshold to judge whether a prediction is positive.
            class2type_map: [optional] dict {class_int:class_name}
        """
        self.ap_iou_thresh = ap_iou_thresh
        self.class2type_map = class2type_map
        self.reset()
        
    def step(self, batch_pred_map_cls, batch_gt_map_cls):
        """ Accumulate one batch of prediction and groundtruth.
        
        Args:
            batch_pred_map_cls: a list of lists [[(pred_cls, pred_box_params, score),...],...]
            batch_gt_map_cls: a list of lists [[(gt_cls, gt_box_params),...],...]
                should have the same length with batch_pred_map_cls (batch_size)
        """
        
        bsize = len(batch_pred_map_cls)
        assert(bsize == len(batch_gt_map_cls))
        for i in range(bsize):
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i] 
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i] 
            self.scan_cnt += 1
    
    def compute_metrics(self):
        """ Use accumulated predictions and groundtruths to compute Average Precision.
        """
        rec, prec, ap = eval_det_multiprocessing(self.pred_map_cls, self.gt_map_cls, ovthresh=self.ap_iou_thresh, get_iou_func=get_iou_obb)
        ret_dict = {} 
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            ret_dict['%s Average Precision'%(clsname)] = ap[key]
        ret_dict['mAP'] = np.mean(list(ap.values()))
        rec_list = []
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            try:
                ret_dict['%s Recall'%(clsname)] = rec[key][-1]
                rec_list.append(rec[key][-1])
            except:
                ret_dict['%s Recall'%(clsname)] = 0
                rec_list.append(0)
        ret_dict['AR'] = np.mean(rec_list)
        return ret_dict

    def reset(self):
        self.gt_map_cls = {} # {scan_id: [(classname, bbox)]}
        self.pred_map_cls = {} # {scan_id: [(classname, bbox, score)]}
        self.scan_cnt = 0


