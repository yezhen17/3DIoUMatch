""" Labeled and unlabeled dataset for 3DIoUMatch

Author: Zhao Na, 2019
Modified by Yezhen Cong, 2020
"""

import os
import sys
import random
import numpy as np
from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
import sunrgbd_utils
from model_util_sunrgbd import SunrgbdDatasetConfig

DC = SunrgbdDatasetConfig() # dataset specific config
MAX_NUM_OBJ = 64 # maximum number of objects allowed per scene
MEAN_COLOR_RGB = np.array([0.5,0.5,0.5]) # sunrgbd color is in 0~1


class SunrgbdSSLLabeledDataset(Dataset):

    def __init__(self, labeled_sample_list=None, num_points=20000, use_color=False, use_height=False,  use_v1=False,
                 augment=False):

        print('--------- Sunrgbd Labeled Dataset Initialization ---------')
        if use_v1:
            self.data_path = os.path.join(ROOT_DIR, 'sunrgbd/sunrgbd_pc_bbox_votes_50k_v1_train')
        else:
            self.data_path = os.path.join(ROOT_DIR, 'sunrgbd/sunrgbd_pc_bbox_votes_50k_v2_train')

        if labeled_sample_list is not None:
            self.scan_names = [x.strip() for x in open(
                os.path.join(ROOT_DIR, 'sunrgbd/sunrgbd_trainval', labeled_sample_list)).readlines()]
            print('\tGet {} labeled scans'.format(len(self.scan_names)))
        else:
            print('Unknown labeled sample list: %s. Exiting...' %labeled_sample_list)
            exit(-1)

        self.num_points = num_points
        self.use_color = use_color
        self.use_height = use_height
        self.augment = augment

    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        """
         Returns a dict with following keys:
             point_clouds: (N,3+C)
             center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
             heading_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
             heading_residual_label: (MAX_NUM_OBJ,)
             size_classe_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
             size_residual_label: (MAX_NUM_OBJ,3)
             sem_cls_label: (MAX_NUM_OBJ,) semantic class index
             box_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
             vote_label: (N,9) with votes XYZ (3 votes: X1Y1Z1, X2Y2Z2, X3Y3Z3)
                 if there is only one vote than X1==X2==X3 etc.
             vote_label_mask: (N,) with 0/1 with 1 indicating the point
                 is in one of the object's OBB.
             scan_idx: int scan index in scan_names list
         """
        scan_name = self.scan_names[idx]
        point_cloud = np.load(os.path.join(self.data_path, scan_name) + '_pc.npz')['pc']  # Nx6
        bboxes = np.load(os.path.join(self.data_path, scan_name) + '_bbox.npy')  # K,8
        point_votes = np.load(os.path.join(self.data_path, scan_name) + '_votes.npz')['point_votes']  # Nx10

        if not self.use_color:
            point_cloud = point_cloud[:, 0:3]
        else:
            point_cloud = point_cloud[:, 0:6]
            point_cloud[:, 3:] = (point_cloud[:, 3:] - MEAN_COLOR_RGB)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)  # (N,4) or (N,7)

        ema_point_cloud = pc_util.random_sampling(point_cloud, self.num_points, return_choices=False)

        # ------------------------------- DATA AUGMENTATION ------------------------------
        flip_x_axis = 0
        flip_y_axis = 0
        rot_mat = np.identity(3)
        scale_ratio = np.ones((1,3))
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                flip_x_axis = 1
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                bboxes[:, 0] = -1 * bboxes[:, 0]
                bboxes[:, 6] = np.pi - bboxes[:, 6]
                point_votes[:, [1, 4, 7]] = -1 * point_votes[:, [1, 4, 7]]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
            rot_mat = sunrgbd_utils.rotz(rot_angle)

            point_votes_end = np.zeros_like(point_votes)
            point_votes_end[:, 1:4] = np.dot(point_cloud[:, 0:3] + point_votes[:, 1:4], np.transpose(rot_mat))
            point_votes_end[:, 4:7] = np.dot(point_cloud[:, 0:3] + point_votes[:, 4:7], np.transpose(rot_mat))
            point_votes_end[:, 7:10] = np.dot(point_cloud[:, 0:3] + point_votes[:, 7:10], np.transpose(rot_mat))

            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            bboxes[:, 0:3] = np.dot(bboxes[:, 0:3], np.transpose(rot_mat))
            bboxes[:, 6] -= rot_angle
            point_votes[:, 1:4] = point_votes_end[:, 1:4] - point_cloud[:, 0:3]
            point_votes[:, 4:7] = point_votes_end[:, 4:7] - point_cloud[:, 0:3]
            point_votes[:, 7:10] = point_votes_end[:, 7:10] - point_cloud[:, 0:3]

            # Augment point cloud scale: 0.85x-1.15x
            scale_ratio = np.random.random() * 0.3 + 0.85
            scale_ratio = np.expand_dims(np.tile(scale_ratio, 3), 0)
            point_cloud[:, 0:3] *= scale_ratio
            bboxes[:, 0:3] *= scale_ratio
            bboxes[:, 3:6] *= scale_ratio
            point_votes[:, 1:4] *= scale_ratio
            point_votes[:, 4:7] *= scale_ratio
            point_votes[:, 7:10] *= scale_ratio
            if self.use_height:
                point_cloud[:, -1] *= scale_ratio[0, 0]

        # ------------------------------- LABELS ------------------------------
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))

        target_bboxes_mask[0:bboxes.shape[0]] = 1
        target_bboxes[0:bboxes.shape[0], :] = bboxes[:, 0:6]

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            semantic_class = bbox[7]
            angle_class, angle_residual = DC.angle2class(bbox[6])
            # NOTE: The mean size stored in size2class is of full length of box edges,
            # while in sunrgbd_data.py data dumping we dumped *half* length l,w,h.. so have to time it by 2 here
            box3d_size = bbox[3:6] * 2
            size_class, size_residual = DC.size2class(box3d_size, DC.class2type[semantic_class])
            angle_classes[i] = angle_class
            angle_residuals[i] = angle_residual
            size_classes[i] = size_class
            size_residuals[i] = size_residual
            target_bboxes_semcls[i] = semantic_class

        point_cloud, choices = pc_util.random_sampling(point_cloud, self.num_points, return_choices=True)
        point_votes_mask = point_votes[choices, 0]
        point_votes = point_votes[choices, 1:]

        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['center_label'] = target_bboxes.astype(np.float32)[:,0:3]
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['size_class_label'] = size_classes.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32)
        ret_dict['vote_label'] = point_votes.astype(np.float32)
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)

        ret_dict['supervised_mask'] = np.array(1).astype(np.int64)
        ret_dict['ema_point_clouds'] = ema_point_cloud.astype(np.float32)
        ret_dict['flip_x_axis'] = np.array(flip_x_axis).astype(np.int64)
        ret_dict['flip_y_axis'] =  np.array(flip_y_axis).astype(np.int64)
        ret_dict['rot_mat'] = rot_mat.astype(np.float32)
        ret_dict['rot_angle'] = np.array(rot_angle).astype(np.float32)
        ret_dict['scale'] = np.array(scale_ratio).astype(np.float32)

        return ret_dict


class SunrgbdSSLUnlabeledDataset(Dataset):
    def __init__(self, labeled_sample_list=None, num_points=20000, use_color=False, use_height=False, use_v1=False,
                 aug_num=1, scan_idx_list=None, load_labels=None):
        print('----------------Sunrgbd Unlabeled Dataset Initialization----------------')
        if use_v1:
            self.data_path = os.path.join(ROOT_DIR, 'sunrgbd/sunrgbd_pc_bbox_votes_50k_v1_train')
        else:
            self.data_path = os.path.join(ROOT_DIR, 'sunrgbd/sunrgbd_pc_bbox_votes_50k_v2_train')

        train_scan_names = sorted(list(set([os.path.basename(x)[0:6] for x in os.listdir(self.data_path)])))
        if scan_idx_list is not None:
            train_scan_names = [self.scan_names[i] for i in scan_idx_list]

        if labeled_sample_list is not None:
            labeled_scan_names = [x.strip() for x in open(
                os.path.join(ROOT_DIR, 'sunrgbd/sunrgbd_trainval', labeled_sample_list)).readlines()]
            if len(train_scan_names) == len(labeled_scan_names):
                self.scan_names = train_scan_names
            else:
                self.scan_names = list(set(train_scan_names) - set(labeled_scan_names))
            print('\tGet {} unlabeled scans out of {}'.format(len(self.scan_names), len(train_scan_names)))
        else:
            print('\tUnknown labeled sample list: %s. Exiting...' % labeled_sample_list)
            exit(-1)

        self.num_points = num_points
        self.use_color = use_color
        self.use_height = use_height
        self.aug_num = aug_num
        self.load_labels = load_labels
        if load_labels:
            print('Warning! Loading labels for analysis')

    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            point_clouds: (N,3+C)
            scan_idx: int scan index in scan_names list
        """
        scan_name = self.scan_names[idx]
        mesh_vertices = np.load(os.path.join(self.data_path, scan_name) + '_pc.npz')['pc']  # Nx6

        if not self.use_color:
            raw_point_cloud = mesh_vertices[:, 0:3]  # do not use color for now
        else:
            raw_point_cloud = mesh_vertices[:, 0:6]
            raw_point_cloud[:, 3:] = (raw_point_cloud[:, 3:] - MEAN_COLOR_RGB) / 256.0

        if self.use_height:
            floor_height = np.percentile(raw_point_cloud[:, 2], 0.99)
            height = raw_point_cloud[:, 2] - floor_height
            raw_point_cloud = np.concatenate([raw_point_cloud, np.expand_dims(height, 1)], 1)

        ret_dict = {}
        ema_point_cloud = pc_util.random_sampling(raw_point_cloud, self.num_points, return_choices=False)
        ret_dict['ema_point_clouds'] = ema_point_cloud.astype(np.float32)

        bboxes = np.load(os.path.join(self.data_path, scan_name) + '_bbox.npy')  # K,8
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))

        target_bboxes_mask[0:bboxes.shape[0]] = 1
        target_bboxes[0:bboxes.shape[0], :] = bboxes[:, 0:6]

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            semantic_class = bbox[7]
            angle_class, angle_residual = DC.angle2class(bbox[6])
            # NOTE: The mean size stored in size2class is of full length of box edges,
            # while in sunrgbd_data.py data dumping we dumped *half* length l,w,h.. so have to time it by 2 here
            box3d_size = bbox[3:6] * 2
            size_class, size_residual = DC.size2class(box3d_size, DC.class2type[semantic_class])
            angle_classes[i] = angle_class
            angle_residuals[i] = angle_residual
            size_classes[i] = size_class
            size_residuals[i] = size_residual
            target_bboxes_semcls[i] = semantic_class

        if self.load_labels:
            ret_dict['center_label'] = target_bboxes.astype(np.float32)[:, 0:3]
            ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
            ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
            ret_dict['size_class_label'] = size_classes.astype(np.int64)
            ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
            ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
            ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32)

        point_cloud, choices = pc_util.random_sampling(raw_point_cloud, self.num_points, return_choices=True)
        flip_x_axis = 0
        flip_y_axis = 0
        rot_angle = 0
        rot_mat = np.identity(3)
        scale_ratio = np.ones((1, 3))
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                flip_x_axis = 1
                point_cloud[:, 0] = -1 * point_cloud[:, 0]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
            rot_mat = pc_util.rotz(rot_angle)
            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))

            # Augment point cloud scale: 0.85x-1.15x
            scale_ratio = np.random.random() * 0.3 + 0.85
            scale_ratio = np.expand_dims(np.tile(scale_ratio, 3), 0)
            point_cloud[:, 0:3] *= scale_ratio
            if self.use_height:
                point_cloud[:, -1] *= scale_ratio[0, 0]

        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['flip_x_axis'] = np.array(flip_x_axis).astype(np.int64)
        ret_dict['flip_y_axis'] = np.array(flip_y_axis).astype(np.int64)
        ret_dict['rot_mat'] = rot_mat.astype(np.float32)
        ret_dict['rot_angle'] = np.array(rot_angle).astype(np.float32)
        ret_dict['scale'] = np.array(scale_ratio).astype(np.float32)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['supervised_mask'] = np.array(0).astype(np.int64)
        return ret_dict