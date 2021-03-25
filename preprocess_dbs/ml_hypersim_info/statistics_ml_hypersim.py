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


def get_task_split_paths(dataset_path, splits_csv_path, split_name, folder_str,
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

    n_cameras = 0
    n_samples = 0
    paths = []
    scenes = df['scene_name'].unique()
    #scenes = scenes[0:60]
    #scenes = scenes[0:150]
    scenes = scenes[0:75]
    for scene in scenes:
        df_scene = df[df['scene_name'] == scene]
        cameras = df_scene['camera_name'].unique()

        # select only test
        if (len(cameras) > 2):
            #cameras = cameras[0:-1]
            cameras = cameras[-1:]
        else:
            cameras = []

        # select only train
        '''
        if (len(cameras) > 2):
            cameras = cameras[0:-1]
        '''
        n_cameras += len(cameras)
        for camera in cameras:
            df_camera = df_scene[df_scene['camera_name'] == camera]
            frames = df_camera['frame_id'].unique()

            n_samples += len(frames)
            for frame in frames:
                path = '%s/%s/images/scene_%s_%s/frame.%04d.%s.%s' % (
                    dataset_path, scene, camera, folder_str, int(frame),
                    task_str, ext)
                paths.append(path)
    print('n cameras %d -- n samples %d' % (n_cameras, n_samples))
    import pdb
    pdb.set_trace()
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


csv_path = '/data/multi-domain-graph-6/datasets/hypersim/metadata_images_split_scene_v1_selection.csv'
get_task_split_paths(dataset_path, csv_path, 'train1', 'final_preview',
                     'tonemap', 'jpg')

df = pd.read_csv(csv_path)
df_frames_included = df[df['included_in_public_release'] == True]

print("# initial frames %d -- # released frames %d" %
      (len(df), len(df_frames_included)))

df = df_frames_included
df_train = df[df['split_partition_name'] == 'train']
df_test = df[df['split_partition_name'] == 'test']
df_val = df[df['split_partition_name'] == 'val']
import pdb
pdb.set_trace()
print('# train frames %d' % len(df_train))
print('# test frames %d' % len(df_test))
print('# val frames %d' % len(df_val))

print('# scenes %d' % len(df['scene_name'].unique()))
print('max nr cameras per scene %d' % (len(df['camera_name'].unique())))
print('max frames per camera %d' % (np.max(df['frame_id'])))

all_scenes = df['scene_name'].unique()
n_frames = []
for scene in all_scenes:
    df_scene = df[df['scene_name'] == scene]
    valid = df_scene[df_scene['included_in_public_release'] == True]
    n_frames.append(len(valid))

n_frames = np.array(n_frames)
print(np.max(n_frames))
pos = np.argwhere(n_frames == np.max(n_frames))[0][0]

print('smallest scene %s' % (all_scenes[pos]))

main_path = r'/data/multi-domain-graph-6/datasets/ml_hypersim'
# scenes/images/scene_cam_00_geometry_hdf5


class DepthDataset(Dataset):
    def __init__(self, main_path):
        super(DepthDataset, self).__init__()
        self.all_frames_paths = sorted(
            glob.glob('%s/*/images/*geometry_hdf5/frame*depth_meters.hdf5' %
                      (main_path)))

    def __getitem__(self, index):
        depth_file = h5py.File(self.all_frames_paths[index], "r")
        depth_info = np.array(depth_file.get('dataset')).astype('float32')
        depth_info = torch.from_numpy(depth_info).unsqueeze(0)
        depth_info = torch.nn.functional.interpolate(depth_info[None],
                                                     (256, 256))[0]
        return depth_info, self.all_frames_paths[index]

    def __len__(self):
        return len(self.all_frames_paths)


depth_dataset = DepthDataset(main_path)
dataloader = DataLoader(depth_dataset,
                        batch_size=100,
                        shuffle=False,
                        num_workers=20)
import pdb
pdb.set_trace()
min_values = []
max_values = []
for batch in tqdm(dataloader):
    depth_info, paths = batch
    nan_mask = depth_info != depth_info
    non_nan_mask = ~nan_mask
    min_values.append(torch.min(depth_info[non_nan_mask]))
    max_values.append(torch.max(depth_info[non_nan_mask]))

    if torch.max(depth_info[non_nan_mask] > 1800):
        import pdb
        pdb.set_trace()
        print(paths)

min_val = np.min(np.array(min_values))
max_val = np.max(np.array(max_values))
print('Depth: min %8.4f -- max %8.4f' % (min_val, max_val))
