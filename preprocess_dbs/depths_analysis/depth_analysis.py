import os
import sys
import shutil
import cv2
import torch
import glob
import h5py
import pandas as pd
from tqdm import tqdm
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import experts.depth_expert
depth_expert = experts.depth_expert.DepthModelXTC(full_expert=True)

WORKING_H = 256
WORKING_W = 256

taskonomy_splits = [
    'tiny-train-0.15-part1', 'tiny-train-0.15-part2', 'tiny-train-0.15-part3',
    'tiny-val', 'tiny-test'
]
# splits: tiny-val, tiny-test, tiny-train-0.15-part1, tiny-train-0.15-part2, tiny-train-0.15-part3
taskonomy_gt_path = r'/data/multi-domain-graph-6/datasets/taskonomy/taskonomy_info'
# /split_name/depth_zbuffer/*.png
taskonomy_rgb_path = r'/data/multi-domain-graph-2/datasets/Taskonomy'
alter_taskonomy_rgb_path = r'/data/multi-domain-graph-6/datasets/taskonomy/taskonomy_info'
# /split_name/rgb/*.png'
taskonomy_gt_th_5 = 400
taskonomy_gt_th_95 = 3404
taskonomy_exp_th_5 = 0.0530
taskonomy_exp_th_95 = 0.4310
taskonomy_gt_th_50 = 0
taskonomy_exp_th_50 = 0

replica_splits = ['train', 'valid', 'test']
replica_gt_path = r'/data/multi-domain-graph-6/datasets/replica_raw'
# /split_name/depth/*.npy
replica_rgb_path = r'/data/multi-domain-graph-6/datasets/replica_raw'
# /split_name/rgb/*.npy
replica_gt_th_5 = 0.469
replica_gt_th_95 = 3.563
replica_exp_th_5 = 0.045
replica_exp_th_95 = 0.24
replica_gt_th_50 = 0.336
replica_exp_th_50 = 0.277

hypersim_splits = ['train1', 'train2', 'train3', 'valid', 'test']
hypersim_db_path = r'/data/multi-domain-graph-6/datasets/hypersim/data'
# /scene_name/images/scene_cam_camIndex_geometry_hdf5/*depth_meters.hdf5
# /scene_name/images/scene_cam_camIndex_final_preview/*tonemap.jpg
hypersim_splits_csv_path = r'/data/multi-domain-graph-6/datasets/hypersim/metadata_images_split_scene_v1_selection.csv'
hypersim_gt_th_5 = 1.155
hypersim_gt_th_95 = 22.33
hypersim_exp_th_5 = 0.05
hypersim_exp_th_95 = 0.342
hypersim_gt_th_50 = 0.166
hypersim_exp_th_50 = 0.289


def hypersim_get_task_split_paths(dataset_path, splits_csv_path, split_name,
                                  folder_str, task_str, ext):
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


class Hypersim_RGB_and_Depth_DB(Dataset):
    def __init__(self, dataset_path, splits_csv_path, splits):
        super(Hypersim_RGB_and_Depth_DB, self).__init__()
        self.rgb_paths = []
        self.depth_paths = []
        for split_name in splits:
            self.rgb_paths = self.rgb_paths + hypersim_get_task_split_paths(
                dataset_path, splits_csv_path, split_name, 'final_preview',
                'tonemap', 'jpg')
            self.depth_paths = self.depth_paths + hypersim_get_task_split_paths(
                dataset_path, splits_csv_path, split_name, 'geometry_hdf5',
                'depth_meters', 'hdf5')

        assert (len(self.depth_paths) == len(self.rgb_paths))

    def __getitem__(self, index):
        rgb_info = cv2.imread(self.rgb_paths[index])
        rgb_info = cv2.resize(rgb_info, (WORKING_W, WORKING_H),
                              cv2.INTER_CUBIC)
        rgb_info = cv2.cvtColor(rgb_info, cv2.COLOR_BGR2RGB)
        rgb_info = rgb_info.astype('float32')

        depth_file = h5py.File(self.depth_paths[index], "r")
        depth_info = np.array(depth_file.get('dataset')).astype('float32')
        depth_info = torch.from_numpy(depth_info).unsqueeze(0)
        depth_info = torch.nn.functional.interpolate(depth_info[None],
                                                     (WORKING_H, WORKING_W))[0]
        #depth_info = depth_info[:, None, :, :]
        #depth_info[depth_info != depth_info] = 0  # get rid of nan values
        #depth_info = depth_info / 31.25
        #depth_info = torch.clamp(depth_info, 0, 1)
        return rgb_info, depth_info

    def __len__(self):
        return len(self.rgb_paths)


class Replica_RGB_and_Depth_DB(Dataset):
    def __init__(self, depth_path, rgb_path, splits):
        super(Replica_RGB_and_Depth_DB, self).__init__()
        self.rgb_paths = []
        self.depth_paths = []
        for split_name in splits:
            if split_name == 'valid':
                split_name = 'val'
            glob_pattern = '%s/%s/depth/*.npy' % (depth_path, split_name)
            self.depth_paths = self.depth_paths + sorted(
                glob.glob(glob_pattern))
            glob_pattern = '%s/%s/rgb/*.npy' % (rgb_path, split_name)
            self.rgb_paths = self.rgb_paths + sorted(glob.glob(glob_pattern))

        assert (len(self.depth_paths) == len(self.rgb_paths))

    def __getitem__(self, index):
        rgb = np.load(self.rgb_paths[index])
        rgb = np.moveaxis(rgb, 0, -1)
        rgb = rgb * 255

        depth = np.load(self.depth_paths[index])
        depth = depth[None]
        return rgb, depth

    def __len__(self):
        return len(self.rgb_paths)


class Taskonomy_RGB_and_Depth_DB(Dataset):
    def __init__(self, depth_path, rgb_path, splits):
        super(Taskonomy_RGB_and_Depth_DB, self).__init__()
        self.depth_paths = []
        self.rgb_paths = []
        #splits = splits[4:5]
        for split_name in splits:
            glob_pattern = '%s/%s/depth_zbuffer/*.png' % (depth_path,
                                                          split_name)
            self.depth_paths = self.depth_paths + sorted(
                glob.glob(glob_pattern))

            glob_pattern = '%s/%s/rgb/*.png' % (rgb_path, split_name)
            self.rgb_paths = self.rgb_paths + sorted(glob.glob(glob_pattern))

        #self.rgb_paths = self.rgb_paths[0:100]
        #self.depth_paths = self.depth_paths[0:100]

        assert (len(self.depth_paths) == len(self.rgb_paths))

    def __getitem__(self, index):

        try:
            rgb = cv2.imread(self.rgb_paths[index])
            rgb = cv2.resize(rgb, (WORKING_W, WORKING_H), cv2.INTER_CUBIC)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        except:
            print(self.rgb_paths[index])
            rgb = np.zeros((WORKING_W, WORKING_H, 3), dtype=np.uint8)

        depth = Image.open(self.depth_paths[index])
        depth = np.array(depth)
        depth = torch.from_numpy(depth[None]).float()
        depth = torch.nn.functional.interpolate(depth[None],
                                                (WORKING_H, WORKING_W))[0]
        return rgb, depth

    def __len__(self):
        return len(self.depth_paths)


class TransFct_Scale():
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def apply(self, data):
        data = data * self.scale_factor
        return data


class TransFct_HistoClamp():
    def __init__(self, th_5, th_95):
        self.th_5 = th_5
        self.th_95 = th_95

    def apply(self, data):
        data = data - self.th_5
        data = data / (self.th_95 - self.th_5)
        return data


class TransFct_HistoHalfClamp():
    def __init__(self, th_95):
        self.th_95 = th_95

    def apply(self, data):
        data = data / self.th_95
        return data


class TransFct_Clamp():
    def __init__(self, min_v, max_v):
        self.min_v = min_v
        self.max_v = max_v

    def apply(self, data):
        data[data < self.min_v] = self.min_v
        data[data > self.max_v] = self.max_v
        return data


class TransFct_Id():
    def __init__(self):
        self.val = 0

    def apply(self, data):
        return data


def get_limits(dataloader, bm_fct, gt_transformations, exp_transformations):
    min_values_gt = []
    max_values_gt = []
    min_values_exp = []
    max_values_exp = []

    l1_fct = torch.nn.L1Loss()
    l2_fct = torch.nn.MSELoss()

    l1 = 0
    l2 = 0

    for batch in tqdm(dataloader):
        rgb, depth = batch
        bm = bm_fct(depth)
        for gt_trans in gt_transformations:
            depth = gt_trans.apply(depth)

        depth_exp = depth_expert.apply_expert_batch(rgb)
        for exp_trans in exp_transformations:
            depth_exp = exp_trans.apply(depth_exp)

        min_values_gt.append(torch.min(depth[bm]))
        max_values_gt.append(torch.max(depth[bm]))
        min_values_exp.append(np.min(depth_exp))
        max_values_exp.append(np.max(depth_exp))

        depth = depth.numpy()
        bm = bm.numpy()
        depth[depth != depth] = 0
        depth = depth * bm
        depth_exp = depth_exp * bm
        depth = torch.from_numpy(depth)
        depth_exp = torch.from_numpy(depth_exp)
        l1 += l1_fct(depth, depth_exp).item()
        l2 += l2_fct(depth, depth_exp).item()

    gt_min = np.min(np.array(min_values_gt))
    gt_max = np.max(np.array(max_values_gt))
    exp_min = np.min(np.array(min_values_exp))
    exp_max = np.max(np.array(max_values_exp))

    l1 = l1 / len(dataloader)
    l2 = l2 / len(dataloader)
    l1 = l1 * 100
    l2 = l2 * 100

    return gt_min, gt_max, exp_min, exp_max, l1, l2


def write_histo_data(out_path, suffix, n_bins, gt_bins, exp_bins, common_bins,
                     gt_histo, cum_gt_histo, exp_histo, cum_exp_histo,
                     common_gt_histo, cum_common_gt_histo, common_exp_histo,
                     cum_common_exp_histo):
    csv_file = open(out_path, 'w')
    csv_file.write(
        'gt_bin_min_val, gt_bin_max_val, histo_gt_%s, cum_histo_gt_%s, exp_bin_min_val, exp_bin_max_val, histo_exp_%s,cum_histo_exp_%s, c_bin_min_val, c_bin_max_val, c_histo_gt_%s, c_cum_histo_gt_%s, c_histo_exp_%s, c_cum_histo_exp_%s\n'
        % (suffix, suffix, suffix, suffix, suffix, suffix, suffix, suffix))
    for i in range(n_bins):
        csv_file.write(
            '%8.4f, %8.4f, %8.4f, %8.4f,' %
            (gt_bins[i], gt_bins[i + 1], gt_histo[i], cum_gt_histo[i]))
        csv_file.write(
            '%8.4f, %8.4f, %8.4f, %8.4f,' %
            (exp_bins[i], exp_bins[i + 1], exp_histo[i], cum_exp_histo[i]))
        csv_file.write('%8.4f, %8.4f, ' % (common_bins[i], common_bins[i + 1]))
        csv_file.write('%8.4f, %8.4f, ' %
                       (common_gt_histo[i], cum_common_gt_histo[i]))
        csv_file.write('%8.4f, %8.4f, ' %
                       (common_exp_histo[i], cum_common_exp_histo[i]))
        csv_file.write('\n')
    csv_file.close()


def save_example(dataloader, bm_fct, save_path, gt_transformations,
                 exp_transformations):

    for batch in tqdm(dataloader):
        rgb, depth = batch
        init_depth = depth[3, 0, :, :]
        bm = bm_fct(depth)
        for gt_trans in gt_transformations:
            depth = gt_trans.apply(depth)
        after_depth = depth[3, 0, :, :]

        depth_exp = depth_expert.apply_expert_batch(rgb)
        init_depth_exp = depth_exp[3, 0, :, :]
        for exp_trans in exp_transformations:
            depth_exp = exp_trans.apply(depth_exp)
        after_depth_exp = depth_exp[3, 0, :, :]

        depth = depth.numpy()
        img_0 = np.concatenate((init_depth, init_depth_exp), 1)
        img_1 = np.concatenate((after_depth, after_depth_exp), 1)
        img = np.concatenate((img_0, img_1), 0)
        cv2.imwrite(save_path, np.uint8(img * 255))
        break


def get_histo(dataloader, bm_fct, gt_min, gt_max, exp_min, exp_max, out_path,
              suffix, quantiles, gt_transformations, exp_transformations):

    n_bins = 10000
    exp_histo = np.zeros(n_bins)
    gt_histo = np.zeros(n_bins)
    common_exp_histo = np.zeros(n_bins)
    common_gt_histo = np.zeros(n_bins)
    min_bin = min(gt_min, exp_min)
    max_bin = max(gt_max, exp_max)
    for batch in tqdm(dataloader):
        rgb, depth = batch
        bm = bm_fct(depth)
        for gt_trans in gt_transformations:
            depth = gt_trans.apply(depth)

        depth_exp = depth_expert.apply_expert_batch(rgb)
        for exp_trans in exp_transformations:
            depth_exp = exp_trans.apply(depth_exp)

        depth = depth.numpy()

        exp_histo_, exp_bins = np.histogram(depth_exp,
                                            bins=n_bins,
                                            range=(exp_min, exp_max))
        exp_histo = exp_histo + exp_histo_

        gt_histo_, gt_bins = np.histogram(depth[bm],
                                          bins=n_bins,
                                          range=(gt_min, gt_max))
        gt_histo = gt_histo + gt_histo_

        exp_histo_, common_bins = np.histogram(depth_exp,
                                               bins=n_bins,
                                               range=(min_bin, max_bin))
        common_exp_histo = common_exp_histo + exp_histo_

        gt_histo_, common_bins = np.histogram(depth[bm],
                                              bins=n_bins,
                                              range=(min_bin, max_bin))
        common_gt_histo = common_gt_histo + gt_histo_

    gt_histo = gt_histo / np.sum(gt_histo)
    cum_gt_histo = np.cumsum(gt_histo)
    exp_histo = exp_histo / np.sum(exp_histo)
    cum_exp_histo = np.cumsum(exp_histo)

    gt_quantiles = []
    exp_quantiles = []
    for quant in quantiles:
        pos = np.argwhere(cum_gt_histo >= quant)[0]
        gt_quantiles.append(gt_bins[pos])

        pos = np.argwhere(cum_exp_histo >= quant)[0]
        exp_quantiles.append(exp_bins[pos])

    common_gt_histo = common_gt_histo / np.sum(common_gt_histo)
    cum_common_gt_histo = np.cumsum(common_gt_histo)
    common_exp_histo = common_exp_histo / np.sum(common_exp_histo)
    cum_common_exp_histo = np.cumsum(common_exp_histo)

    write_histo_data(out_path, suffix, n_bins, gt_bins, exp_bins, common_bins,
                     gt_histo, cum_gt_histo, exp_histo, cum_exp_histo,
                     common_gt_histo, cum_common_gt_histo, common_exp_histo,
                     cum_common_exp_histo)

    return gt_quantiles, exp_quantiles


def get_medians_and_save_histo(dataloader, bm_fct, gt_min, gt_max, exp_min,
                               exp_max, out_path, suffix):
    n_bins = 10000
    exp_histo = np.zeros(n_bins)
    gt_histo = np.zeros(n_bins)
    common_exp_histo = np.zeros(n_bins)
    common_gt_histo = np.zeros(n_bins)
    min_bin = min(gt_min, exp_min)
    max_bin = max(gt_max, exp_max)
    for batch in tqdm(dataloader):
        rgb, depth = batch
        depth_exp = depth_expert.apply_expert_batch(rgb)
        bm = bm_fct(depth)

        depth = depth.numpy()

        exp_histo_, exp_bins = np.histogram(depth_exp,
                                            bins=n_bins,
                                            range=(exp_min, exp_max))
        exp_histo = exp_histo + exp_histo_

        gt_histo_, gt_bins = np.histogram(depth[bm],
                                          bins=n_bins,
                                          range=(gt_min, gt_max))
        gt_histo = gt_histo + gt_histo_

        exp_histo_, common_bins = np.histogram(depth_exp,
                                               bins=n_bins,
                                               range=(min_bin, max_bin))
        common_exp_histo = common_exp_histo + exp_histo_

        gt_histo_, common_bins = np.histogram(depth[bm],
                                              bins=n_bins,
                                              range=(min_bin, max_bin))
        common_gt_histo = common_gt_histo + gt_histo_

    gt_histo = gt_histo / np.sum(gt_histo)
    cum_gt_histo = np.cumsum(gt_histo)
    exp_histo = exp_histo / np.sum(exp_histo)
    cum_exp_histo = np.cumsum(exp_histo)

    pos = np.argwhere(cum_gt_histo >= 0.5)[0]
    gt_th_50 = gt_bins[pos]

    pos = np.argwhere(cum_exp_histo >= 0.5)[0]
    exp_th_50 = exp_bins[pos]

    common_gt_histo = common_gt_histo / np.sum(common_gt_histo)
    cum_common_gt_histo = np.cumsum(common_gt_histo)
    common_exp_histo = common_exp_histo / np.sum(common_exp_histo)
    cum_common_exp_histo = np.cumsum(common_exp_histo)

    write_histo_data(out_path, suffix, gt_bins, exp_bins, common_bins,
                     gt_histo, cum_gt_histo, exp_histo, cum_exp_histo,
                     common_gt_histo, cum_common_gt_histo, common_exp_histo,
                     cum_common_exp_histo)

    return gt_th_50, exp_th_50


def get_after_scale_5_and_95_limits_and_save_histo(dataloader, bm_fct,
                                                   s_gt_min, s_gt_max,
                                                   s_exp_min, s_exp_max,
                                                   scale_factor_for_exp,
                                                   out_path, suffix):
    n_bins = 10000
    exp_histo = np.zeros(n_bins)
    gt_histo = np.zeros(n_bins)
    common_exp_histo = np.zeros(n_bins)
    common_gt_histo = np.zeros(n_bins)
    min_bin = min(s_gt_min, s_exp_min)
    max_bin = max(s_gt_max, s_exp_max)

    for batch in tqdm(dataloader):
        rgb, depth = batch
        depth_exp = depth_expert.apply_expert_batch(rgb)
        depth_exp = depth_exp * scale_factor_for_exp
        bm = bm_fct(depth)

        depth = depth.numpy()

        exp_histo_, exp_bins = np.histogram(depth_exp,
                                            bins=n_bins,
                                            range=(exp_min, exp_max))
        exp_histo = exp_histo + exp_histo_

        gt_histo_, gt_bins = np.histogram(depth[bm],
                                          bins=n_bins,
                                          range=(gt_min, gt_max))
        gt_histo = gt_histo + gt_histo_

        exp_histo_, common_bins = np.histogram(depth_exp,
                                               bins=n_bins,
                                               range=(min_bin, max_bin))
        common_exp_histo = common_exp_histo + exp_histo_

        gt_histo_, common_bins = np.histogram(depth[bm],
                                              bins=n_bins,
                                              range=(min_bin, max_bin))
        common_gt_histo = common_gt_histo + gt_histo_

    gt_histo = gt_histo / np.sum(gt_histo)
    cum_gt_histo = np.cumsum(gt_histo)
    exp_histo = exp_histo / np.sum(exp_histo)
    cum_exp_histo = np.cumsum(exp_histo)

    pos = np.argwhere(cum_gt_histo >= 0.05)[0]
    gt_th_5 = gt_bins[pos]
    pos = np.argwhere(cum_gt_histo >= 0.95)[0]
    gt_th_95 = gt_bins[pos]

    pos = np.argwhere(cum_exp_histo >= 0.05)[0]
    exp_th_5 = exp_bins[pos]
    pos = np.argwhere(cum_exp_histo >= 0.95)[0]
    exp_th_95 = exp_bins[pos]

    common_gt_histo = common_gt_histo / np.sum(common_gt_histo)
    cum_common_gt_histo = np.cumsum(common_gt_histo)
    common_exp_histo = common_exp_histo / np.sum(common_exp_histo)
    cum_common_exp_histo = np.cumsum(common_exp_histo)

    write_histo_data(out_path, suffix, gt_bins, exp_bins, common_bins,
                     gt_histo, cum_gt_histo, exp_histo, cum_exp_histo,
                     common_gt_histo, cum_common_gt_histo, common_exp_histo,
                     cum_common_exp_histo)

    return gt_th_5, gt_th_95, exp_th_5, exp_th_95


def get_histo_after_scale_and_norm(dataloader, bm_fct, sn_gt_min, sn_gt_max,
                                   sn_exp_min, sn_exp_max, gt_th_5, gt_th_95,
                                   exp_th_5, exp_th_95, scale_factor_for_exp,
                                   csv_path, suffix):
    n_bins = 10000
    exp_histo = np.zeros(n_bins)
    gt_histo = np.zeros(n_bins)
    common_exp_histo = np.zeros(n_bins)
    common_gt_histo = np.zeros(n_bins)
    min_bin = min(sn_gt_min, sn_exp_min)
    max_bin = max(sn_gt_max, sn_exp_max)
    for batch in tqdm(dataloader):
        rgb, depth = batch
        bm = bm_fct(depth)
        depth_exp = depth_expert.apply_expert_batch(rgb)
        depth_exp = depth_exp * scale_factor_for_exp

        depth_exp = (depth_exp - exp_th_5) / (exp_th_95 - exp_th_5)
        depth = (depth - gt_th_5) / (gt_th_95 - gt_th_5)

        depth = depth.numpy()

        exp_histo_, exp_bins = np.histogram(depth_exp,
                                            bins=n_bins,
                                            range=(sn_exp_min, sn_exp_max))
        exp_histo = exp_histo + exp_histo_

        gt_histo_, gt_bins = np.histogram(depth[bm],
                                          bins=n_bins,
                                          range=(sn_gt_min, sn_gt_max))
        gt_histo = gt_histo + gt_histo_

    gt_histo = gt_histo / np.sum(gt_histo)
    cum_gt_histo = np.cumsum(gt_histo)
    exp_histo = exp_histo / np.sum(exp_histo)
    cum_exp_histo = np.cumsum(exp_histo)

    csv_file = open(csv_path, 'w')
    csv_file.write(
        'gt_bin_min_val, gt_bin_max_val, histo_gt_%s, cum_histo_gt_%s, exp_bin_min_val, exp_bin_max_val, histo_exp_%s,cum_histo_exp_%s,\n'
        % (suffix, suffix, suffix, suffix))
    for i in range(n_bins):
        csv_file.write(
            '%8.4f, %8.4f, %8.4f, %8.4f,' %
            (gt_bins[i], gt_bins[i + 1], gt_histo[i], cum_gt_histo[i]))
        csv_file.write(
            '%8.4f, %8.4f, %8.4f, %8.4f,' %
            (exp_bins[i], exp_bins[i + 1], exp_histo[i], cum_exp_histo[i]))
        csv_file.write('\n')
    csv_file.close()


def taskonomy_get_mask_of_valid_samples(data):
    bm = data < 65535
    return bm


def replica_get_mask_of_valid_samples(data):
    bm = data > 0
    return bm


def hypersim_get_mask_of_valid_samples(data):
    bm = data != data
    bm = ~bm
    return bm


def post_process_data(data, th_5, th_95):
    data = data - th_5
    data = data / (th_95 - th_5)
    return data


if __name__ == "__main__":
    argv = sys.argv
    run_type = np.int32(
        argv[1])  # which step to enable - if (-1) => perform all
    dataset = argv[2]
    splits = argv[3:]

    if dataset == 'taskonomy':
        db_type = Taskonomy_RGB_and_Depth_DB
        gt_path = taskonomy_gt_path
        rgb_path = taskonomy_rgb_path
        if splits[0] == 'all':
            splits = taskonomy_splits
        bm_fct = taskonomy_get_mask_of_valid_samples
    if dataset == 'replica':
        db_type = Replica_RGB_and_Depth_DB
        gt_path = replica_gt_path
        rgb_path = replica_rgb_path
        if splits[0] == 'all':
            splits = replica_splits
        bm_fct = replica_get_mask_of_valid_samples
    if dataset == 'hypersim':
        db_type = Hypersim_RGB_and_Depth_DB
        gt_path = hypersim_db_path
        rgb_path = hypersim_splits_csv_path
        if splits[0] == 'all':
            splits = hypersim_splits
        bm_fct = hypersim_get_mask_of_valid_samples
        max_gt_val = 770

    if argv[3] == 'all':
        print('%s' % (dataset))
        db = db_type(gt_path, rgb_path, splits)
        dataloader = DataLoader(db,
                                batch_size=30,
                                shuffle=False,
                                num_workers=20)
        # Step 0 - get ranges for both gt and expert results & errors
        gt_min, gt_max, exp_min, exp_max, l1, l2 = get_limits(
            dataloader, bm_fct, [], [])
        print('GT  min: %8.4f  --  max: %8.4f' % (gt_min, gt_max))
        print('EXP min: %8.4f  --  max: %8.4f' % (exp_min, exp_max))
        print('L1 : %8.4f' % (l1))
        print('L2 : %8.4f' % (l2))

        # Step 1 - get medians for gt and expert results & save histograms
        csv_path = '%s_initial_histo_run_%d.csv' % (dataset, run_type)
        suffix = '%s_all' % (dataset)
        gt_quants, exp_quants = get_histo(dataloader, bm_fct, gt_min, gt_max,
                                          exp_min, exp_max, csv_path, suffix,
                                          np.array([0.5]), [], [])
        gt_th_50 = gt_quants[0]
        exp_th_50 = exp_quants[0]
        print('GT  th_50: %8.4f' % (gt_th_50))
        print('EXP th_50: %8.4f' % (exp_th_50))
        scale_factor_for_exp = gt_th_50 / exp_th_50
        print('Scale factor for EXP : %8.4f' % (scale_factor_for_exp))

        # Step 2 - get range for scaled exp & errors
        exp_scale_trans = TransFct_Scale(scale_factor_for_exp)

        s_gt_min, s_gt_max, s_exp_min, s_exp_max, s_l1, s_l2 = get_limits(
            dataloader, bm_fct, [], [exp_scale_trans])
        print('after scale: GT  min: %8.4f  --  max: %8.4f' %
              (s_gt_min, s_gt_max))
        print('after scale: EXP min: %8.4f  --  max: %8.4f' %
              (s_exp_min, s_exp_max))
        print('after scale L1 : %8.4f' % (s_l1))
        print('after scale L2 : %8.4f' % (s_l2))

        # Step 3 - get histo of scaled exp
        csv_path = '%s_scale_exp_histo_run_%d.csv' % (dataset, run_type)
        suffix = '%s_all' % (dataset)
        gt_quants, exp_quants = get_histo(dataloader, bm_fct, s_gt_min,
                                          s_gt_max, s_exp_min, s_exp_max,
                                          csv_path, suffix,
                                          np.array([0.05, 0.95]), [],
                                          [exp_scale_trans])
        gt_th_5 = gt_quants[0]
        gt_th_95 = gt_quants[1]
        exp_th_5 = exp_quants[0]
        exp_th_95 = exp_quants[1]
        print('GT  th_5: %8.4f -- th_95: %8.4f' % (gt_th_5, gt_th_95))
        print('EXP th_5: %8.4f -- th_95: %8.4f' % (exp_th_5, exp_th_95))

        # Step 4 - get range for scaled exp & norm both & errors
        if run_type == 0 or run_type == 1:
            exp_norm_trans = TransFct_HistoClamp(exp_th_5, exp_th_95)
            gt_norm_trans = TransFct_HistoClamp(gt_th_5, gt_th_95)
        else:
            exp_norm_trans = TransFct_HistoHalfClamp(exp_th_95)
            gt_norm_trans = TransFct_HistoHalfClamp(gt_th_95)

        if run_type == 1 or run_type == 3:
            exp_clamp = TransFct_Clamp(0, 1)
        else:
            exp_clamp = TransFct_Id()

        sn_gt_min, sn_gt_max, sn_exp_min, sn_exp_max, sn_l1, sn_l2 = get_limits(
            dataloader, bm_fct, [gt_norm_trans],
            [exp_scale_trans, exp_norm_trans, exp_clamp])
        print('after scale & norm: GT  min: %8.4f  --  max: %8.4f' %
              (sn_gt_min, sn_gt_max))
        print('after scale & norm: EXP min: %8.4f  --  max: %8.4f' %
              (sn_exp_min, sn_exp_max))
        print('after scale & norm L1 : %8.4f' % (sn_l1))
        print('after scale & norm L2 : %8.4f' % (sn_l2))

        # Step 5 - get histo of scaled exp & norm both + eval
        csv_path = '%s_scale_exp_norm_all_histo_run_%d.csv' % (dataset,
                                                               run_type)
        suffix = '%s_all' % (dataset)
        _ = get_histo(dataloader, bm_fct, sn_gt_min, sn_gt_max, sn_exp_min,
                      sn_exp_max, csv_path, suffix, [], [gt_norm_trans],
                      [exp_scale_trans, exp_norm_trans, exp_clamp])

        # Save example
        save_path = '%s_example_run_%d.png' % (dataset, run_type)
        save_example(dataloader, bm_fct, save_path, [gt_norm_trans],
                     [exp_scale_trans, exp_norm_trans, exp_clamp])