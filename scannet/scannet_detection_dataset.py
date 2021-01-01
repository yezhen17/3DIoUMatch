# coding: utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Modified by Zhao Na, 2019
# Modified by Yezhen Cong, 2020

""" Dataset for object bounding box regression.
An axis aligned bounding box is parameterized by (cx,cy,cz) and (dx,dy,dz)
where (cx,cy,cz) is the center point of the box, dx is the x-axis length of the box.
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
from model_util_scannet import rotate_aligned_boxes

from model_util_scannet import ScannetDatasetConfig
DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 64
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

class ScannetDetectionDataset(Dataset):
       
    def __init__(self, split_set='train', labeled_ratio=0.1, labeled_sample_list=None, num_points=20000,
                        use_color=False, use_height=False, augment=False, remove_obj=False, test_transductive=False):

        print('--------- DetectionDataset ', split_set, ' Initialization ---------')
        self.data_path = os.path.join(BASE_DIR, 'scannet_train_detection_data')
        all_scan_names = list(set([os.path.basename(x)[0:12] \
            for x in os.listdir(self.data_path) if x.startswith('scene')]))
        if split_set=='all':            
            self.scan_names = all_scan_names
        elif split_set in ['train', 'val', 'test']:
            split_filenames = os.path.join(ROOT_DIR, 'scannet/meta_data',
                'scannetv2_{}.txt'.format(split_set))
            with open(split_filenames, 'r') as f:
                self.scan_names = f.read().splitlines()   
            # remove unavailiable scans
            num_scans = len(self.scan_names)
            self.scan_names = [sname for sname in self.scan_names \
                if sname in all_scan_names]
            print('\tkept {} scans out of {}'.format(len(self.scan_names), num_scans))
            num_scans = len(self.scan_names)
        else:
            print('\tillegal split name')
            return
        
        self.num_points = num_points
        self.use_color = use_color        
        self.use_height = use_height
        self.augment = augment
        self.remove_obj = remove_obj

        # added
        self.raw_data_path = os.path.join(ROOT_DIR, 'scannet/meta_data')

        # construct labeled and unlabeled samples for training
        if split_set == 'train':
            if test_transductive:
                if labeled_sample_list is not None:
                    labeled_scan_names = [x.strip() for x in open(
                        os.path.join(self.raw_data_path, labeled_sample_list)).readlines()]
                    self.scan_names = list(set(self.scan_names) - set(labeled_scan_names))
                    print('\tGet {} unlabeled scans for transductive learning'.format(len(self.scan_names)))
                else:
                    print('Unknown labeled sample list: %s. Exiting...' % labeled_sample_list)
                    exit(-1)
            else:
                self.labeled_ratio = labeled_ratio
                self.labeled_sample_list = labeled_sample_list
                self.get_labeled_samples()
       
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
            pcl_color: unused
        """
        
        scan_name = self.scan_names[idx]
        mesh_vertices = np.load(os.path.join(self.data_path, scan_name)+'_vert.npy')
        instance_labels = np.load(os.path.join(self.data_path, scan_name)+'_ins_label.npy')
        semantic_labels = np.load(os.path.join(self.data_path, scan_name)+'_sem_label.npy')
        instance_bboxes = np.load(os.path.join(self.data_path, scan_name)+'_bbox.npy')

        if self.remove_obj:
            if np.random.random() > 0.5 and instance_bboxes.shape[0]>=3:
                #random remove an object
                removed_box_ind = random.choice(list(range(0, instance_bboxes.shape[0])))
                # NOTE: this assumes obj_id is in 1,2,3,.,,,.NUM_INSTANCES
                removed_obj_ind = removed_box_ind + 1
                removed_verts_inds = np.where(instance_labels==removed_obj_ind)[0]

                instance_bboxes = np.delete(instance_bboxes, removed_box_ind, axis=0)
                mesh_vertices = np.delete(mesh_vertices, removed_verts_inds, axis=0)
                instance_labels = np.delete(instance_labels, removed_verts_inds, axis=0)
                semantic_labels = np.delete(semantic_labels, removed_verts_inds, axis=0)

        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3] # do not use color for now
            pcl_color = mesh_vertices[:,3:6]
        else:
            point_cloud = mesh_vertices[:,0:6] 
            point_cloud[:,3:] = (point_cloud[:,3:]-MEAN_COLOR_RGB)/256.0
        
        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) 
            
        # ------------------------------- LABELS ------------------------------        
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))    
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        point_cloud, choices = pc_util.random_sampling(point_cloud, self.num_points, return_choices=True)

        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        pcl_color = pcl_color[choices]
        
        target_bboxes_mask[0:instance_bboxes.shape[0]] = 1
        target_bboxes[0:instance_bboxes.shape[0],:] = instance_bboxes[:,0:6]
        
        # ------------------------------- DATA AUGMENTATION ------------------------------        
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:,0] = -1 * point_cloud[:,0]
                target_bboxes[:,0] = -1 * target_bboxes[:,0]

            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:,1] = -1 * point_cloud[:,1]
                target_bboxes[:,1] = -1 * target_bboxes[:,1]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
            rot_mat = pc_util.rotz(rot_angle)
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes(target_bboxes, rot_mat)

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
                x = point_cloud[ind,:3]
                center = 0.5*(x.min(0) + x.max(0))
                point_votes[ind, :] = center - x
                point_votes_mask[ind] = 1.0
        point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical 
        
        class_ind = [np.where(DC.nyu40ids == x)[0][0] for x in instance_bboxes[:,-1]]   
        # NOTE: set size class as semantic class. Consider use size2class.
        size_classes[0:instance_bboxes.shape[0]] = class_ind
        size_residuals[0:instance_bboxes.shape[0], :] = \
            target_bboxes[0:instance_bboxes.shape[0], 3:6] - DC.mean_size_arr[class_ind,:]

        target_bboxes_semcls[0:instance_bboxes.shape[0]] = class_ind
            
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
        ret_dict['pcl_color'] = pcl_color
        ret_dict['supervised_mask'] = np.array(1).astype(np.int64)

        scene_label = np.zeros(DC.num_class)
        unique_class_ind = list(set(class_ind))
        for ind in unique_class_ind:
            scene_label[int(ind)] = 1
        ret_dict['scene_label'] = scene_label.astype(np.float32)
        return ret_dict


    def get_labeled_samples(self):
        if self.labeled_sample_list is not None:
            labeled_scan_names = [x.strip() for x in open(
                os.path.join(ROOT_DIR, 'scannet/meta_data', self.labeled_sample_list)).readlines()]
        else:
            # randomly select scan names w.r.t labeled_ratio
            num_scans = len(self.scan_names)
            num_labeled_scans = int(self.labeled_ratio * num_scans)
            scan2label = np.zeros((num_scans, DC.num_class))
            for i, scan_name in enumerate(self.scan_names):
                instance_bboxes = np.load(os.path.join(self.data_path, scan_name) + '_bbox.npy')
                class_ind = [DC.nyu40id2class[x] for x in instance_bboxes[:, -1]]
                if class_ind != []:
                    unique_class_ind = list(set(class_ind))
                else: continue
                for j in unique_class_ind:
                    scan2label[i,j] = 1

            while True:
                choices = np.random.choice(num_scans, num_labeled_scans, replace=False)
                class_distr = np.sum(scan2label[choices], axis=0)
                class_mask = np.where(class_distr>0, 1, 0)
                if np.sum(class_mask) == DC.num_class:
                    labeled_scan_names = list(np.array(self.scan_names)[choices])
                    with open(os.path.join(ROOT_DIR, 'scannet/meta_data/scannetv2_train_{}.txt'.format(self.labeled_ratio)), 'w') as f:
                        for scan_name in labeled_scan_names:
                            f.write(scan_name + '\n')
                    break

        unlabeled_scan_names = list(set(self.scan_names) - set(labeled_scan_names))
        print('\tSelected {} labeled scans, remained {} unlabeled scans'.format(len(labeled_scan_names), len(unlabeled_scan_names)))
        self.scan_names = labeled_scan_names
        print('first 3 scans', self.scan_names[:3])
        
############# Visualizaion ########

def viz_votes(pc, point_votes, point_votes_mask, name=''):
    """ Visualize point votes and point votes mask labels
    pc: (N,3 or 6), point_votes: (N,9), point_votes_mask: (N,)
    """
    inds = (point_votes_mask==1)
    pc_obj = pc[inds,0:3]
    pc_obj_voted1 = pc_obj + point_votes[inds,0:3]    
    pc_util.write_ply(pc_obj, 'pc_obj{}.ply'.format(name))
    pc_util.write_ply(pc_obj_voted1, 'pc_obj_voted1{}.ply'.format(name))
    
def viz_obb(pc, label, mask, angle_classes, angle_residuals,
    size_classes, size_residuals, name=''):
    """ Visualize oriented bounding box ground truth
    pc: (N,3)
    label: (K,3)  K == MAX_NUM_OBJ
    mask: (K,)
    angle_classes: (K,)
    angle_residuals: (K,)
    size_classes: (K,)
    size_residuals: (K,3)
    """
    oriented_boxes = []
    K = label.shape[0]
    for i in range(K):
        if mask[i] == 0: continue
        obb = np.zeros(7)
        obb[0:3] = label[i,0:3]
        heading_angle = 0 # hard code to 0
        box_size = DC.mean_size_arr[size_classes[i], :] + size_residuals[i, :]
        obb[3:6] = box_size
        obb[6] = -1 * heading_angle
        print(obb)        
        oriented_boxes.append(obb)
    pc_util.write_oriented_bbox(oriented_boxes, 'gt_obbs{}.ply'.format(name))
    pc_util.write_ply(label[mask==1,:], 'gt_centroids{}.ply'.format(name))

    
if __name__=='__main__': 
    dset = ScannetDetectionDataset(use_height=True, num_points=40000)
    for i_example in range(4):
        example = dset.__getitem__(1)
        pc_util.write_ply(example['point_clouds'], 'pc_{}.ply'.format(i_example))    
        viz_votes(example['point_clouds'], example['vote_label'],
            example['vote_label_mask'],name=i_example)    
        viz_obb(pc=example['point_clouds'], label=example['center_label'],
            mask=example['box_label_mask'],
            angle_classes=None, angle_residuals=None,
            size_classes=example['size_class_label'], size_residuals=example['size_residual_label'],
            name=i_example)
