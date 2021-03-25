import os
import shutil
import sys
import h5py
import torch
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import glob
from torch.utils.data import DataLoader, Dataset

dataset_path = r'/data/multi-domain-graph-6/datasets/hypersim/data'
csv_path = '/data/multi-domain-graph-6/datasets/hypersim/metadata_images_split_scene_v1_selection.csv'


def get_original_paths(dataset_path, splits_csv_path, split_name, folder_str,
                       task_str, ext):

    if split_name == "valid":
        split_name = "val"
    train_index = -1
    if split_name.find('train') != -1:
        train_index = int(split_name[-1])
        split_name = 'train'

    df = pd.read_csv(splits_csv_path)
    df = df[df['included_in_public_release'] == True]
    df = df[df['split_partition_name'] == split_name]

    paths = []
    scenes = df['scene_name'].unique()
    for scene in scenes:
        df_scene = df[df['scene_name'] == scene]
        cameras = df_scene['camera_name'].unique()

        for camera in cameras:
            df_camera = df_scene[df_scene['camera_name'] == camera]
            frames = df_camera['frame_id'].unique()

            for frame in frames:
                path = '%s/%s/images/scene_%s_%s/frame.%04d.%s.%s' % (
                    dataset_path, scene, camera, folder_str, int(frame),
                    task_str, ext)
                paths.append(path)
    if train_index >= 0:
        n_samples = len(paths)
        n_set_samples = n_samples // 3
        first = n_set_samples + (n_samples - 3 * n_set_samples)
        second = first + n_set_samples
        third = second + n_set_samples
        if train_index == 1:
            paths = paths[0:first]
        elif train_index == 2:
            paths = paths[first:second]
        elif train_index == 3:
            paths = paths[second:third]
        else:
            paths = []
    return paths


def get_v2_task_split_paths(dataset_path, splits_csv_path, split_name,
                            folder_str, task_str, ext):

    initial_split_name = 'train'
    df = pd.read_csv(splits_csv_path)
    df = df[df['included_in_public_release'] == True]
    df = df[df['split_partition_name'] == initial_split_name]

    train_index = -1
    if split_name.find('train') != -1:
        train_index = int(split_name[-1])
        split_name = 'train'
    paths = []
    scenes = df['scene_name'].unique()
    scenes = scenes[0:150]
    camera_index = -1
    for scene in scenes:
        df_scene = df[df['scene_name'] == scene]
        cameras = df_scene['camera_name'].unique()

        if split_name == 'train':
            if (len(cameras) > 2):
                cameras = cameras[0:-1]
        else:
            if (len(cameras) > 2):
                cameras = cameras[-1:]
            else:
                cameras = []

        for camera in cameras:
            camera_index += 1
            if split_name == 'train' and train_index == 1 and camera_index % 2 == 1:
                continue
            if split_name == 'train' and train_index == 2 and camera_index % 2 == 0:
                continue
            if split_name == 'test' and camera_index % 4 == 0:
                continue
            if split_name == 'valid' and camera_index % 4 > 0:
                continue
            df_camera = df_scene[df_scene['camera_name'] == camera]
            frames = df_camera['frame_id'].unique()
            for frame in frames:
                path = '%s/%s/images/scene_%s_%s/frame.%04d.%s.%s' % (
                    dataset_path, scene, camera, folder_str, int(frame),
                    task_str, ext)
                paths.append(path)
    return paths


def get_original_indexes(new_paths, old_paths_list):
    split_indexes = []
    frames_indexes = []
    for path in new_paths:
        for i in range(len(old_paths_list)):
            old_paths = old_paths_list[i]
            if path in old_paths:
                split_indexes.append(i)
                frames_indexes.append(old_paths.index(path))
    return split_indexes, frames_indexes


def copy_data(out_path, in_paths, split_indexes, frames_indexes, dom_name):
    for i in range(len(frames_indexes)):
        split_idx = split_indexes[i]
        frame_idx = frames_indexes[i]
        in_path_ = os.path.join(in_paths[split_idx], dom_name,
                                '%08d.npy' % frame_idx)
        out_path_ = os.path.join(out_path, '%08d.npy' % i)
        shutil.copyfile(in_path_, out_path_)


def process_data(domains, in_paths, out_path):
    for dom in domains:
        print(dom)
        # train1
        print('train1')
        train1_out_path = os.path.join(out_path, 'train1', dom)
        os.makedirs(train1_out_path, exist_ok=True)
        copy_data(train1_out_path, in_paths, train1_split_indexes,
                  train1_frames_indexes, dom)

        # train2
        print('train2')
        train2_out_path = os.path.join(out_path, 'train2', dom)
        os.makedirs(train2_out_path, exist_ok=True)
        copy_data(train2_out_path, in_paths, train2_split_indexes,
                  train2_frames_indexes, dom)

        # test
        print('test')
        test_out_path = os.path.join(out_path, 'test', dom)
        os.makedirs(test_out_path, exist_ok=True)
        copy_data(test_out_path, in_paths, test_split_indexes,
                  test_frames_indexes, dom)

        # valid
        print('valid')
        valid_out_path = os.path.join(out_path, 'valid', dom)
        os.makedirs(valid_out_path, exist_ok=True)
        copy_data(valid_out_path, in_paths, valid_split_indexes,
                  valid_frames_indexes, dom)


#print('GT')
#process_data(gt_domains, gt_in_paths, gt_out_path)
#print('EXP')
#process_data(exp_domains, exp_in_paths, exp_out_path)

if __name__ == '__main__':

    orig_train1_paths = get_original_paths(dataset_path, csv_path, 'train1',
                                           'final_preview', 'tonemap', 'jpg')
    orig_train2_paths = get_original_paths(dataset_path, csv_path, 'train2',
                                           'final_preview', 'tonemap', 'jpg')
    orig_train3_paths = get_original_paths(dataset_path, csv_path, 'train3',
                                           'final_preview', 'tonemap', 'jpg')
    orig_test_paths = get_original_paths(dataset_path, csv_path, 'test',
                                         'final_preview', 'tonemap', 'jpg')
    orig_valid_paths = get_original_paths(dataset_path, csv_path, 'valid',
                                          'final_preview', 'tonemap', 'jpg')

    train1_paths = get_v2_task_split_paths(dataset_path, csv_path, 'train1',
                                           'final_preview', 'tonemap', 'jpg')
    train2_paths = get_v2_task_split_paths(dataset_path, csv_path, 'train2',
                                           'final_preview', 'tonemap', 'jpg')
    test_paths = get_v2_task_split_paths(dataset_path, csv_path, 'test',
                                         'final_preview', 'tonemap', 'jpg')
    valid_paths = get_v2_task_split_paths(dataset_path, csv_path, 'valid',
                                          'final_preview', 'tonemap', 'jpg')

    print('train1 -- %d' % len(train1_paths))
    print('train2 -- %d' % len(train2_paths))
    print('test -- %d' % len(test_paths))
    print('valid -- %d' % len(valid_paths))

    # get indexes of train
    train1_split_indexes, train1_frames_indexes = get_original_indexes(
        train1_paths,
        [orig_train1_paths, orig_train2_paths, orig_train3_paths])
    train2_split_indexes, train2_frames_indexes = get_original_indexes(
        train2_paths,
        [orig_train1_paths, orig_train2_paths, orig_train3_paths])
    test_split_indexes, test_frames_indexes = get_original_indexes(
        test_paths, [orig_train1_paths, orig_train2_paths, orig_train3_paths])
    valid_split_indexes, valid_frames_indexes = get_original_indexes(
        valid_paths, [orig_train1_paths, orig_train2_paths, orig_train3_paths])

    print('train1 -- %d' % len(train1_split_indexes))
    print('train2 -- %d' % len(train2_split_indexes))
    print('test -- %d' % len(test_split_indexes))
    print('valid -- %d' % len(valid_split_indexes))

    gt_out_path = '/data/multi-domain-graph-5/datasets/datasets_preproc_gt/hypersim_v2'
    exp_out_path = '/data/multi-domain-graph-5/datasets/datasets_preproc_exp/hypersim_v2'
    gt_in_paths = [
        '/data/multi-domain-graph-5/datasets/datasets_preproc_gt/hypersim/train1',
        '/data/multi-domain-graph-5/datasets/datasets_preproc_gt/hypersim/train2',
        '/data/multi-domain-graph-5/datasets/datasets_preproc_gt/hypersim/train3'
    ]
    gt_domains = [
        'depth_n_1', 'grayscale', 'halftone_gray', 'hsv', 'normals', 'rgb'
    ]

    exp_in_paths = [
        '/data/multi-domain-graph-5/datasets/datasets_preproc_exp/hypersim/train1',
        '/data/multi-domain-graph-5/datasets/datasets_preproc_exp/hypersim/train2',
        '/data/multi-domain-graph-5/datasets/datasets_preproc_exp/hypersim/train3'
    ]
    exp_domains = [
        'cartoon_wb', 'depth_n_1_xtc', 'edges_dexined', 'normals_xtc',
        'sem_seg_hrnet', 'sobel_large', 'sobel_small', 'sobel_medium',
        'superpixel_fcn'
    ]

    if sys.argv[1] == 'gt':
        print('GT')
        process_data([sys.argv[2]], gt_in_paths, gt_out_path)
    else:
        print('EXP')
        process_data([sys.argv[2]], exp_in_paths, exp_out_path)
