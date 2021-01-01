""" Generate random data splits

Written by: Yezhen Cong, 2020
"""

import os
import sys
import numpy as np

from scannet.model_util_scannet import ScannetDatasetConfig
from sunrgbd.model_util_sunrgbd import SunrgbdDatasetConfig

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR


def gen_scannet_split(labeled_ratio, count):
    DC = ScannetDatasetConfig()
    split_set = 'train'
    split_filenames = os.path.join(ROOT_DIR, 'scannet/meta_data',
                                   'scannetv2_{}.txt'.format(split_set))
    with open(split_filenames, 'r') as f:
        scan_names = f.read().splitlines()
        # remove unavailiable scans
    num_scans = len(scan_names)
    scan2label = np.zeros((num_scans, DC.num_class))
    num_labeled_scans = int(labeled_ratio * num_scans)
    data_path = os.path.join(BASE_DIR, 'scannet/scannet_train_detection_data')
    for i, scan_name in enumerate(scan_names):
        instance_bboxes = np.load(os.path.join(data_path, scan_name) + '_bbox.npy')
        class_ind = [DC.nyu40id2class[x] for x in instance_bboxes[:, -1]]
        if class_ind != []:
            unique_class_ind = list(set(class_ind))
        else:
            continue
        for j in unique_class_ind:
            scan2label[i, j] = 1

    while True:
        choices = np.random.choice(num_scans, num_labeled_scans, replace=False)
        class_distr = np.sum(scan2label[choices], axis=0)
        class_mask = np.where(class_distr > 0, 1, 0)
        if np.sum(class_mask) == DC.num_class:
            labeled_scan_names = list(np.array(scan_names)[choices])
            with open(os.path.join(ROOT_DIR, 'scannet/meta_data/scannetv2_train_{}_{}.txt'.format(labeled_ratio, count)),
                      'w') as f:
                for scan_name in labeled_scan_names:
                    f.write(scan_name + '\n')
            break

    unlabeled_scan_names = list(set(scan_names) - set(labeled_scan_names))
    print('\tSelected {} labeled scans, remained {} unlabeled scans'.format(len(labeled_scan_names), len(unlabeled_scan_names)))


def gen_sunrgbd_split(labeled_ratio, count):
    DC = SunrgbdDatasetConfig()
    data_path = os.path.join(ROOT_DIR, 'sunrgbd/sunrgbd_pc_bbox_votes_50k_v1_%s' % ('train'))
    raw_data_path = os.path.join(ROOT_DIR, 'sunrgbd/sunrgbd_trainval')
    scan_names = sorted(list(set([os.path.basename(x)[0:6] for x in os.listdir(data_path)])))
    num_scans = len(scan_names)
    num_labeled_scans = int(labeled_ratio * num_scans)
    scan2label = np.zeros((num_scans, DC.num_class))
    for i, scan_name in enumerate(scan_names):
        bboxes = np.load(os.path.join(data_path, scan_name) + '_bbox.npy')  # K,8
        class_ind = bboxes[:, -1]
        if len(class_ind) != 0:
            unique_class_ind = np.unique(class_ind)
        else:
            continue
        for j in unique_class_ind:
            scan2label[i, int(j)] = 1

    while True:
        choices = np.random.choice(num_scans, num_labeled_scans, replace=False)
        class_distr = np.sum(scan2label[choices], axis=0)
        class_mask = np.where(class_distr > 0, 1, 0)
        if np.sum(class_mask) == DC.num_class:
            labeled_scan_names = list(np.array(scan_names)[choices])
            with open(os.path.join(raw_data_path, 'sunrgbd_v1_train_{}_{}.txt'.format(labeled_ratio, count)), 'w') as f:
                for scan_name in labeled_scan_names:
                    f.write(scan_name + '\n')
            break

    unlabeled_scan_names = list(set(scan_names) - set(labeled_scan_names))
    print('Selected {} labeled scans, remained {} unlabeled scans'.format(len(labeled_scan_names), len(unlabeled_scan_names)))

ratio = float(sys.argv[1])
dataset = sys.argv[2]
count = int(sys.argv[3])

if dataset == 'scannet':
    gen_scannet_split(ratio, count)
else:
    gen_sunrgbd_split(ratio, count)
