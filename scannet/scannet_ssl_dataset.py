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
import utils.pc_util as pc_util
from scannet.model_util_scannet import ScannetDatasetConfig, rotate_aligned_boxes

DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 64
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])


class ScannetSSLLabeledDataset(Dataset):

    def __init__(self, labeled_sample_list=None, num_points=20000, use_color=False, use_height=False, augment=False):

        print('--------- Scannet Labeled Dataset Initialization ---------')
        self.data_path = os.path.join(BASE_DIR, 'scannet_train_detection_data')
        if labeled_sample_list is not None:
            self.scan_names = [x.strip() for x in open(
                os.path.join(ROOT_DIR, 'scannet/meta_data', labeled_sample_list)).readlines()]
            print('\tGet {} labeled scans'.format(len(self.scan_names)))
            print('first 3 scans', self.scan_names[:3])
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
            sem_cls_label: (MAX_NUM_OBJ,) semantic class index
            angle_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
            angle_residual_label: (MAX_NUM_OBJ,)
            size_classe_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
            size_residual_label: (MAX_NUM_OBJ,3)
            box_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
            point_votes: (N,3) with votes XYZ
            point_votes_mask: (N,) with 0/1 with 1 indicating the point is in one of the object's OBB.
            scan_idx: int scan index in scan_names list
        """

        scan_name = self.scan_names[idx]
        mesh_vertices = np.load(os.path.join(self.data_path, scan_name) + '_vert.npy')
        instance_labels = np.load(os.path.join(self.data_path, scan_name) + '_ins_label.npy')
        semantic_labels = np.load(os.path.join(self.data_path, scan_name) + '_sem_label.npy')
        instance_bboxes = np.load(os.path.join(self.data_path, scan_name) + '_bbox.npy')

        if not self.use_color:
            raw_point_cloud = mesh_vertices[:, 0:3]  # do not use color for now
        else:
            raw_point_cloud = mesh_vertices[:, 0:6]
            raw_point_cloud[:, 3:] = (raw_point_cloud[:, 3:] - MEAN_COLOR_RGB) / 256.0

        if self.use_height:
            floor_height = np.percentile(raw_point_cloud[:, 2], 0.99)
            height = raw_point_cloud[:, 2] - floor_height
            raw_point_cloud = np.concatenate([raw_point_cloud, np.expand_dims(height, 1)], 1)

            # ------------------------------- LABELS ------------------------------
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))

        point_cloud, choices = pc_util.random_sampling(raw_point_cloud, self.num_points, return_choices=True)
        ema_point_cloud = pc_util.random_sampling(raw_point_cloud, self.num_points, return_choices=False)
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]

        target_bboxes_mask[0:instance_bboxes.shape[0]] = 1
        target_bboxes[0:instance_bboxes.shape[0], :] = instance_bboxes[:, 0:6]

        # ------------------------------- DATA AUGMENTATION ------------------------------
        flip_x_axis = 0
        flip_y_axis = 0
        rot_mat = np.identity(3)
        scale_ratio = np.ones((1, 3))
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                flip_x_axis = 1
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                target_bboxes[:, 0] = -1 * target_bboxes[:, 0]

            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                flip_y_axis = 1
                point_cloud[:, 1] = -1 * point_cloud[:, 1]
                target_bboxes[:, 1] = -1 * target_bboxes[:, 1]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
            rot_mat = pc_util.rotz(rot_angle)
            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes(target_bboxes, rot_mat)

            # Augment point cloud scale: 0.85x-1.15x
            scale_ratio = np.random.random() * 0.3 + 0.85
            scale_ratio = np.expand_dims(np.tile(scale_ratio, 3), 0)
            point_cloud[:, 0:3] *= scale_ratio
            target_bboxes[:, 0:3] *= scale_ratio
            target_bboxes[:, 3:6] *= scale_ratio
            if self.use_height:
                point_cloud[:, -1] *= scale_ratio[0, 0]

        # compute votes *AFTER* augmentation
        # generate votes
        # Note: since there's no map between bbox instance labels and
        # pc instance_labels (it had been filtered
        # in the data preparation step) we'll compute the instance bbox
        # from the points sharing the same instance label.
        point_votes = np.zeros([self.num_points, 3])
        point_votes_mask = np.zeros(self.num_points)
        for i_instance in np.unique(instance_labels):
            # find all points belong to that instance
            ind = np.where(instance_labels == i_instance)[0]
            # find the semantic label
            if semantic_labels[ind[0]] in DC.nyu40ids:
                x = point_cloud[ind, :3]
                center = 0.5 * (x.min(0) + x.max(0))
                point_votes[ind, :] = center - x
                point_votes_mask[ind] = 1.0
        point_votes = np.tile(point_votes, (1, 3))  # make 3 votes identical

        class_ind = [np.where(DC.nyu40ids == x)[0][0] for x in instance_bboxes[:, -1]]
        # NOTE: set size class as semantic class. Consider use size2class.
        size_classes[0:instance_bboxes.shape[0]] = class_ind
        size_residuals[0:instance_bboxes.shape[0], :] = \
            target_bboxes[0:instance_bboxes.shape[0], 3:6] - DC.mean_size_arr[class_ind, :]

        target_bboxes_semcls[0:instance_bboxes.shape[0]] = class_ind

        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['center_label'] = target_bboxes.astype(np.float32)[:, 0:3]
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

        scene_label = np.zeros(DC.num_class)
        unique_class_ind = list(set(class_ind))
        for ind in unique_class_ind:
            scene_label[int(ind)] = 1
        ret_dict['scene_label'] = scene_label.astype(np.float32)

        ret_dict['ema_point_clouds'] = ema_point_cloud.astype(np.float32)
        ret_dict['flip_x_axis'] = np.array(flip_x_axis).astype(np.int64)
        ret_dict['flip_y_axis'] =  np.array(flip_y_axis).astype(np.int64)
        ret_dict['rot_mat'] =  rot_mat.astype(np.float32)
        ret_dict['rot_angle'] = np.array(rot_angle).astype(np.float32)
        ret_dict['scale'] = np.array(scale_ratio).astype(np.float32)

        return ret_dict


class ScannetSSLUnlabeledDataset(Dataset):
    def __init__(self, labeled_sample_list=None, num_points=20000, use_color=False,
                 use_height=False, augment=True, load_labels=False):
        print('----------------Scannet Unlabeled Dataset Initialization----------------')
        self.data_path = os.path.join(BASE_DIR, 'scannet_train_detection_data')
        all_scan_names = list(set([os.path.basename(x)[0:12] \
                                   for x in os.listdir(self.data_path) if x.startswith('scene')]))
        split_filenames = os.path.join(ROOT_DIR, 'scannet/meta_data/scannetv2_train.txt')
        with open(split_filenames, 'r') as f:
            train_scan_names = f.read().splitlines()
        # remove unavailiable scans
        train_scan_names = [sname for sname in train_scan_names if sname in all_scan_names]

        if labeled_sample_list is not None:
            labeled_scan_names = [x.strip() for x in open(
                os.path.join(ROOT_DIR, 'scannet/meta_data', labeled_sample_list)).readlines()]
            if len(train_scan_names) == len(labeled_scan_names):
                self.scan_names = train_scan_names
            else:
                self.scan_names = list(set(train_scan_names) - set(labeled_scan_names))
            print('\tGet {} unlabeled scans out of {}'.format(len(self.scan_names), len(train_scan_names)))
        else:
            print('\tUnknown labeled sample list: %s. Exiting...' %labeled_sample_list)
            exit(-1)
        self.scan_names.sort()

        self.num_points = num_points
        self.use_color = use_color
        self.use_height = use_height
        self.augment = augment
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
        mesh_vertices = np.load(os.path.join(self.data_path, scan_name) + '_vert.npy')
        # instance_labels = np.load(os.path.join(self.data_path, scan_name) + '_ins_label.npy')
        # semantic_labels = np.load(os.path.join(self.data_path, scan_name) + '_sem_label.npy')
        instance_bboxes = np.load(os.path.join(self.data_path, scan_name) + '_bbox.npy')

        if not self.use_color:
            raw_point_cloud = mesh_vertices[:, 0:3]  # do not use color for now
        else:
            raw_point_cloud = mesh_vertices[:, 0:6]
            raw_point_cloud[:, 3:] = (raw_point_cloud[:, 3:] - MEAN_COLOR_RGB) / 256.0

        if self.use_height:
            floor_height = np.percentile(raw_point_cloud[:, 2], 0.99)
            height = raw_point_cloud[:, 2] - floor_height
            raw_point_cloud = np.concatenate([raw_point_cloud, np.expand_dims(height, 1)], 1)

        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))

        ret_dict = {}
        ema_point_cloud = pc_util.random_sampling(raw_point_cloud, self.num_points, return_choices=False)
        ret_dict['ema_point_clouds'] = ema_point_cloud.astype(np.float32)

        # instance_labels = instance_labels[choices]
        # semantic_labels = semantic_labels[choices]

        target_bboxes_mask[0:instance_bboxes.shape[0]] = 1
        target_bboxes[0:instance_bboxes.shape[0], :] = instance_bboxes[:, 0:6]
        class_ind = [np.where(DC.nyu40ids == x)[0][0] for x in instance_bboxes[:, -1]]
        # NOTE: set size class as semantic class. Consider use size2class.
        size_classes[0:instance_bboxes.shape[0]] = class_ind
        size_residuals[0:instance_bboxes.shape[0], :] = \
            target_bboxes[0:instance_bboxes.shape[0], 3:6] - DC.mean_size_arr[class_ind, :]

        target_bboxes_semcls[0:instance_bboxes.shape[0]] = class_ind

        if self.load_labels:
            ret_dict['center_label'] = target_bboxes.astype(np.float32)[:, 0:3]
            ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32)
            ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
            ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
            ret_dict['size_class_label'] = size_classes.astype(np.int64)
            ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
            ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)

        point_cloud, choices = pc_util.random_sampling(raw_point_cloud, self.num_points, return_choices=True)
        # ------------------------------- DATA AUGMENTATION ------------------------------
        flip_x_axis = 0
        flip_y_axis = 0
        rot_mat = np.identity(3)
        scale_ratio = np.ones((1, 3))
        rot_angle = 0
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                flip_x_axis = 1
                point_cloud[:, 0] = -1 * point_cloud[:, 0]

            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                flip_y_axis = 1
                point_cloud[:, 1] = -1 * point_cloud[:, 1]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
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
