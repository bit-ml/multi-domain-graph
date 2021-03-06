import os
import sys
import shutil
import cv2
import torch
import glob
import h5py
import matplotlib.pyplot as plt
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

replica_splits = ['train', 'valid', 'test']
replica_gt_path = r'/data/multi-domain-graph-6/datasets/replica_raw_1'
# /split_name/depth/*.npy
replica_rgb_path = r'/data/multi-domain-graph-6/datasets/replica_raw_1'
# /split_name/rgb/*.npy
replica2_gt_path = r'/data/multi-domain-graph-2/datasets/replica_raw_2'
replica2_rgb_path = r'/data/multi-domain-graph-2/datasets/replica_raw_2'

hypersim_splits = ['train1', 'train2', 'train3', 'valid', 'test']
hypersim_db_path = r'/data/multi-domain-graph-6/datasets/hypersim/data'
# /scene_name/images/scene_cam_camIndex_geometry_hdf5/*depth_meters.hdf5
# /scene_name/images/scene_cam_camIndex_final_preview/*tonemap.jpg
hypersim_splits_csv_path = r'/data/multi-domain-graph-6/datasets/hypersim/metadata_images_split_scene_v1_selection.csv'


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
        # self.depth_paths = self.depth_paths[0:100]
        # self.rgb_paths = self.rgb_paths[0:100]

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


class TransFct_ScaleMinMax():
    def __init__(self, min_v, max_v):
        self.min_v = min_v
        self.max_v = max_v

    def apply(self, data):
        data = (data - self.min_v) / (self.max_v - self.min_v)
        return data


class TransFct_Gamma():
    def __init__(self, gamma_factor):
        self.gamma_factor = gamma_factor

    def apply(self, data):
        if torch.is_tensor(data):
            data = torch.pow(data, self.gamma_factor)
        else:
            data = np.power(data, self.gamma_factor)
        return data


class TransFct_Scale():
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def apply(self, data):
        data = data * self.scale_factor
        return data


class TransFct_HistoClamp():
    def __init__(self, th1, th2):
        self.th1 = th1
        self.th2 = th2

    def apply(self, data):
        data = data - self.th1
        data = data / (self.th2 - self.th1)
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


def get_histo_specification(cum_target_histo, cum_data_histo, n_bins):

    cum_target_histo = (cum_target_histo * n_bins).round().astype('int32')
    cum_data_histo = (cum_data_histo * n_bins).round().astype('int32')

    inv_cum_target_histo = np.zeros(n_bins + 1)

    for i in range(n_bins + 1):
        pos = np.argwhere(cum_target_histo >= i)
        inv_cum_target_histo[i] = pos[0]

    cum_data_histo = cum_data_histo.round().astype('int32')
    inv_cum_target_histo = inv_cum_target_histo.round().astype('int32')

    return inv_cum_target_histo, cum_data_histo


class TransFct_HistoSpecification_GT():
    def __init__(self, dataloader, bm_fct, gt_transformations,
                 exp_transformations, n_bins):
        gt_histo = np.zeros(n_bins + 1)
        for batch in tqdm(dataloader):
            rgb, depth = batch
            bm = bm_fct(depth)
            for gt_trans in gt_transformations:
                depth = gt_trans.apply(depth)

            depth = depth.numpy()

            gt_histo_, _ = np.histogram(depth[bm],
                                        bins=n_bins + 1,
                                        range=(0, 1))
            gt_histo = gt_histo + gt_histo_

        gt_histo = gt_histo / np.sum(gt_histo)
        cum_gt_histo = np.cumsum(gt_histo)

        target_histo = gt_histo * 0 + 1
        target_histo = target_histo / np.sum(target_histo)
        cum_target_histo = np.cumsum(target_histo)

        inv_cum_target_histo, cum_data_histo = get_histo_specification(
            cum_target_histo, cum_gt_histo, n_bins)
        self.n_bins = n_bins
        self.cum_data_histo = cum_data_histo
        self.inv_cum_target_histo = inv_cum_target_histo

    def apply(self, data):
        data = data.numpy()
        data_ = data * self.n_bins
        data_ = data_.astype('int32')
        data_ = self.inv_cum_target_histo[self.cum_data_histo[data_]]
        data_ = data_.astype('float32')
        data_ = data_ / self.n_bins
        return torch.tensor(data_)


class TransFct_HistoSpecification_Exp_v2():
    def __init__(self, dataloader, bm_fct, gt_transformations,
                 exp_transformations, n_bins):
        exp_histo = np.zeros(n_bins + 1)
        for batch in tqdm(dataloader):
            rgb, depth = batch
            bm = bm_fct(depth)
            for gt_trans in gt_transformations:
                depth = gt_trans.apply(depth)

            depth_exp = depth_expert.apply_expert_batch(rgb)
            for exp_trans in exp_transformations:
                depth_exp = exp_trans.apply(depth_exp)

            depth = depth.numpy()

            exp_histo_, _ = np.histogram(depth_exp,
                                         bins=n_bins + 1,
                                         range=(0, 1))
            exp_histo = exp_histo + exp_histo_

        exp_histo = exp_histo / np.sum(exp_histo)
        cum_exp_histo = np.cumsum(exp_histo)

        target_histo = exp_histo * 0 + 1
        target_histo = target_histo / np.sum(target_histo)
        cum_target_histo = np.cumsum(target_histo)

        inv_cum_target_histo, cum_data_histo = get_histo_specification(
            cum_target_histo, cum_exp_histo, n_bins)
        self.n_bins = n_bins
        self.cum_data_histo = cum_data_histo
        self.inv_cum_target_histo = inv_cum_target_histo

    def apply(self, data):
        data_ = data * self.n_bins
        data_ = data_.astype('int32')
        data_ = self.inv_cum_target_histo[self.cum_data_histo[data_]]
        data_ = data_.astype('float32')
        data_ = data_ / self.n_bins
        return data_


class TransFct_HistoSpecification_Exp():
    def __init__(self, dataloader, bm_fct, gt_transformations,
                 exp_transformations, n_bins):
        exp_histo = np.zeros(n_bins + 1)
        gt_histo = np.zeros(n_bins + 1)
        for batch in tqdm(dataloader):
            rgb, depth = batch
            bm = bm_fct(depth)
            for gt_trans in gt_transformations:
                depth = gt_trans.apply(depth)

            depth_exp = depth_expert.apply_expert_batch(rgb)
            for exp_trans in exp_transformations:
                depth_exp = exp_trans.apply(depth_exp)

            depth = depth.numpy()

            exp_histo_, _ = np.histogram(depth_exp,
                                         bins=n_bins + 1,
                                         range=(0, 1))
            exp_histo = exp_histo + exp_histo_

            gt_histo_, _ = np.histogram(depth[bm],
                                        bins=n_bins + 1,
                                        range=(0, 1))
            gt_histo = gt_histo + gt_histo_

        gt_histo = gt_histo / np.sum(gt_histo)
        cum_gt_histo = np.cumsum(gt_histo)
        exp_histo = exp_histo / np.sum(exp_histo)
        cum_exp_histo = np.cumsum(exp_histo)

        inv_cum_target_histo, cum_data_histo = get_histo_specification(
            cum_gt_histo, cum_exp_histo, n_bins)
        self.n_bins = n_bins
        self.cum_data_histo = cum_data_histo
        self.inv_cum_target_histo = inv_cum_target_histo

    def apply(self, data):
        data_ = data * self.n_bins
        data_ = data_.astype('int32')
        data_ = self.inv_cum_target_histo[self.cum_data_histo[data_]]
        data_ = data_.astype('float32')
        data_ = data_ / self.n_bins
        return data_


def get_limits(dataloader, bm_fct, gt_transformations, exp_transformations):
    min_values_gt = []
    max_values_gt = []
    min_values_exp = []
    max_values_exp = []

    l1_fct = torch.nn.L1Loss()
    l2_fct = torch.nn.MSELoss()

    l1 = 0
    l2 = 0

    l1_01 = 0
    l2_01 = 0

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

        depth = torch.clamp(depth, 0, 1)
        l1_01 += l1_fct(depth, depth_exp).item()
        l2_01 += l2_fct(depth, depth_exp).item()

    gt_min = np.min(np.array(min_values_gt))
    gt_max = np.max(np.array(max_values_gt))
    exp_min = np.min(np.array(min_values_exp))
    exp_max = np.max(np.array(max_values_exp))

    l1 = l1 / len(dataloader)
    l2 = l2 / len(dataloader)
    l1 = l1 * 100
    l2 = l2 * 100

    l1_01 = l1_01 / len(dataloader)
    l2_01 = l2_01 / len(dataloader)
    l1_01 = l1_01 * 100
    l2_01 = l2_01 * 100

    return gt_min, gt_max, exp_min, exp_max, l1, l2, l1_01, l2_01


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


def write_histo_data_v2(out_path, suffix, n_bins, bins, gt_histo, cum_gt_histo,
                        exp_histo, cum_exp_histo):
    csv_file = open(out_path, 'w')
    csv_file.write(
        'bin_min_val, bin_max_val, histo_gt_%s, cum_histo_gt_%s, histo_exp_%s, cum_histo_exp_%s\n'
        % (suffix, suffix, suffix, suffix))
    for i in range(n_bins):
        csv_file.write('%8.4f, %8.4f,' % (bins[i], bins[i + 1]))
        csv_file.write('%8.4f, %8.4f,' % (gt_histo[i], cum_gt_histo[i]))
        csv_file.write('%8.4f, %8.4f,' % (exp_histo[i], cum_exp_histo[i]))
        csv_file.write('\n')
    csv_file.close()


def save_example(dataloader, bm_fct, save_path, gt_transformations,
                 exp_transformations):

    for batch in tqdm(dataloader):
        rgb, depth = batch

        bm = bm_fct(depth)
        idx = 0
        for gt_trans in gt_transformations:
            depth = gt_trans.apply(depth)
            if idx == 0:
                init_depth = depth[3, 0, :, :]
            idx += 1
        after_depth = depth[3, 0, :, :]

        depth_exp = depth_expert.apply_expert_batch(rgb)

        idx = 0
        for exp_trans in exp_transformations:
            depth_exp = exp_trans.apply(depth_exp)
            if idx == 0:
                init_depth_exp = depth_exp[3, 0, :, :]
            idx += 1
        after_depth_exp = depth_exp[3, 0, :, :]

        depth = depth.numpy()
        img_0 = np.concatenate((init_depth, init_depth_exp), 1)
        img_1 = np.concatenate((after_depth, after_depth_exp), 1)
        img = np.concatenate((img_0, img_1), 0)
        cv2.imwrite(save_path, np.uint8(img * 255))
        break


def get_histo(dataloader, bm_fct, n_bins, gt_min, gt_max, exp_min, exp_max,
              out_path, suffix, quantiles, gt_transformations,
              exp_transformations):
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


def get_histogram(dataloader, bm_fct, n_bins, out_path, suffix,
                  gt_transformations, exp_transformations, min_val, max_val):
    exp_histo = np.zeros(n_bins)
    gt_histo = np.zeros(n_bins)
    for batch in tqdm(dataloader):
        rgb, depth = batch
        bm = bm_fct(depth)
        for gt_trans in gt_transformations:
            depth = gt_trans.apply(depth)

        depth_exp = depth_expert.apply_expert_batch(rgb)
        for exp_trans in exp_transformations:
            depth_exp = exp_trans.apply(depth_exp)

        depth = depth.numpy()

        exp_histo_, histo_bins = np.histogram(depth_exp,
                                              bins=n_bins,
                                              range=(min_val, max_val))
        exp_histo = exp_histo + exp_histo_

        gt_histo_, histo_bins = np.histogram(depth[bm],
                                             bins=n_bins,
                                             range=(min_val, max_val))
        gt_histo = gt_histo + gt_histo_

    gt_histo = gt_histo / np.sum(gt_histo)
    cum_gt_histo = np.cumsum(gt_histo)
    exp_histo = exp_histo / np.sum(exp_histo)
    cum_exp_histo = np.cumsum(exp_histo)

    write_histo_data_v2(out_path, suffix, n_bins, histo_bins, gt_histo,
                        cum_gt_histo, exp_histo, cum_exp_histo)

    return histo_bins, gt_histo, cum_gt_histo, exp_histo, cum_exp_histo


def get_histo_v2(dataloader, bm_fct, n_bins, gt_min, gt_max, exp_min, exp_max,
                 out_path, suffix, quantiles, gt_transformations,
                 exp_transformations):
    exp_histo = np.zeros(n_bins)
    gt_histo = np.zeros(n_bins)
    for batch in tqdm(dataloader):
        rgb, depth = batch
        bm = bm_fct(depth)
        for gt_trans in gt_transformations:
            depth = gt_trans.apply(depth)

        depth_exp = depth_expert.apply_expert_batch(rgb)
        for exp_trans in exp_transformations:
            depth_exp = exp_trans.apply(depth_exp)

        depth = depth.numpy()

        depth = (depth - gt_min) / (gt_max - gt_min)
        depth_exp = (depth_exp - exp_min) / (exp_max - exp_min)

        exp_histo_, histo_bins = np.histogram(depth_exp,
                                              bins=n_bins,
                                              range=(-1, 2))
        exp_histo = exp_histo + exp_histo_

        gt_histo_, histo_bins = np.histogram(depth[bm],
                                             bins=n_bins,
                                             range=(-1, 2))
        gt_histo = gt_histo + gt_histo_

    gt_histo = gt_histo / np.sum(gt_histo)
    cum_gt_histo = np.cumsum(gt_histo)
    exp_histo = exp_histo / np.sum(exp_histo)
    cum_exp_histo = np.cumsum(exp_histo)

    gt_quantiles = []
    exp_quantiles = []
    for quant in quantiles:
        pos = np.argwhere(cum_gt_histo >= quant)[0]
        gt_quantiles.append(histo_bins[pos])

        pos = np.argwhere(cum_exp_histo >= quant)[0]
        exp_quantiles.append(histo_bins[pos])

    write_histo_data_v2(out_path, suffix, n_bins, histo_bins, gt_histo,
                        cum_gt_histo, exp_histo, cum_exp_histo)

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


def run_analysis(dataset, split_name, exp_name, splits, db_type, gt_path,
                 rgb_path, bm_fct, gt_th1, gt_th2):
    n_bins_gen_histo = 1000
    n_bins_histospecification = 100000
    prefix = '%s_%s_%s_%04.2f_%04.2f' % (dataset, split_name, exp_name, gt_th1,
                                         gt_th2)
    print('%s' % (prefix))
    db = db_type(gt_path, rgb_path, splits)
    dataloader = DataLoader(db, batch_size=30, shuffle=False, num_workers=20)
    # Step 0 - get ranges for both gt and expert results & errors
    gt_min, gt_max, exp_min, exp_max, l1, l2, l1_01, l2_01 = get_limits(
        dataloader, bm_fct, [], [])
    print('GT  min: %8.4f  --  max: %8.4f' % (gt_min, gt_max))
    print('EXP min: %8.4f  --  max: %8.4f' % (exp_min, exp_max))
    print('L1 : %8.4f' % (l1))
    print('L2 : %8.4f' % (l2))
    print('L1 [0,1] : %8.4f' % (l1_01))
    print('L2 [0,1] : %8.4f' % (l2_01))

    # Step 1 - scale all in range [0,1]
    gt_scale = TransFct_ScaleMinMax(gt_min, gt_max)
    exp_scale = TransFct_ScaleMinMax(exp_min, exp_max)
    np.save('%s_gt_min.npy' % (prefix), gt_min)
    np.save('%s_gt_max.npy' % (prefix), gt_max)
    np.save('%s_exp_min.npy' % (prefix), exp_min)
    np.save('%s_exp_max.npy' % (prefix), exp_max)

    gt_min, gt_max, exp_min, exp_max, l1, l2, l1_01, l2_01 = get_limits(
        dataloader, bm_fct, [gt_scale], [exp_scale])
    print('GT  min: %8.4f  --  max: %8.4f' % (gt_min, gt_max))
    print('EXP min: %8.4f  --  max: %8.4f' % (exp_min, exp_max))
    print('L1 : %8.4f' % (l1))
    print('L2 : %8.4f' % (l2))
    print('L1 [0,1] : %8.4f' % (l1_01))
    print('L2 [0,1] : %8.4f' % (l2_01))

    # Step 2 - compute histograms of scaled values
    csv_path = './logs_%s/%s_initial_histo.csv' % (dataset, prefix)
    suffix = '%s_all' % (prefix)
    histo_bins, gt_histo, cum_gt_histo, exp_histo, cum_exp_histo = get_histogram(
        dataloader, bm_fct, n_bins_gen_histo, csv_path, suffix, [gt_scale],
        [exp_scale], 0, 1)
    # save plot
    plt.plot(histo_bins[1:], gt_histo, label='gt')
    plt.plot(histo_bins[1:], exp_histo, label='exp')
    plt.legend()
    plt.savefig('./logs_%s/%s_initial_histo.png' % (dataset, prefix))
    plt.close()

    # get gt th for gt_th1 of histogram and gt_th2

    pos = np.argwhere(cum_gt_histo >= gt_th1)[0]
    gt_th1_value = histo_bins[pos]
    pos = np.argwhere(cum_gt_histo >= gt_th2)[0]
    gt_th2_value = histo_bins[pos]
    np.save('%s_gt_th1.npy' % prefix, gt_th1_value)
    np.save('%s_gt_th2.npy' % prefix, gt_th2_value)
    gt_clamp_trans = TransFct_HistoClamp(gt_th1_value, gt_th2_value)

    # Step 3 - get limits & eval for scaled and clamped gt (95%)
    gt_min, gt_max, exp_min, exp_max, l1, l2, l1_01, l2_01 = get_limits(
        dataloader, bm_fct, [gt_scale, gt_clamp_trans], [exp_scale])
    print('GT  min: %8.4f  --  max: %8.4f' % (gt_min, gt_max))
    print('EXP min: %8.4f  --  max: %8.4f' % (exp_min, exp_max))
    print('L1 : %8.4f' % (l1))
    print('L2 : %8.4f' % (l2))
    print('L1 [0,1] : %8.4f' % (l1_01))
    print('L2 [0,1] : %8.4f' % (l2_01))

    csv_path = './logs_%s/%s_afterScale_and_GTClamp_histo.csv' % (dataset,
                                                                  prefix)
    suffix = '%s_all' % (prefix)
    histo_bins, gt_histo, cum_gt_histo, exp_histo, cum_exp_histo = get_histogram(
        dataloader, bm_fct, n_bins_gen_histo, csv_path, suffix,
        [gt_scale, gt_clamp_trans], [exp_scale], 0, 1)
    # save plot
    plt.plot(histo_bins[1:], gt_histo, label='gt')
    plt.plot(histo_bins[1:], exp_histo, label='exp')
    plt.legend()
    plt.savefig('./logs_%s/%s_GTkeep95p_histo.png' % (dataset, prefix))
    plt.close()

    exp_histo_specification = TransFct_HistoSpecification_Exp(
        dataloader, bm_fct, [gt_scale, gt_clamp_trans], [exp_scale],
        n_bins_histospecification)
    np.save('%s_n_bins.npy' % prefix, exp_histo_specification.n_bins)
    np.save('%s_cum_exp_histo.npy' % prefix,
            exp_histo_specification.cum_exp_histo)
    np.save('%s_inv_cum_gt_histo.npy' % prefix,
            exp_histo_specification.inv_cum_gt_histo)
    gt_min, gt_max, exp_min, exp_max, l1, l2, l1_01, l2_01 = get_limits(
        dataloader, bm_fct, [gt_scale, gt_clamp_trans],
        [exp_scale, exp_histo_specification])
    print('GT  min: %8.4f  --  max: %8.4f' % (gt_min, gt_max))
    print('EXP min: %8.4f  --  max: %8.4f' % (exp_min, exp_max))
    print('L1 : %8.4f' % (l1))
    print('L2 : %8.4f' % (l2))
    print('L1 [0,1] : %8.4f' % (l1_01))
    print('L2 [0,1] : %8.4f' % (l2_01))

    csv_path = './logs_%s/%s_final_histo.csv' % (dataset, prefix)
    suffix = '%s_all' % (prefix)
    histo_bins, gt_histo, cum_gt_histo, exp_histo, cum_exp_histo = get_histogram(
        dataloader, bm_fct, n_bins_gen_histo, csv_path, suffix,
        [gt_scale, gt_clamp_trans], [exp_scale, exp_histo_specification], 0, 1)
    # save plot
    plt.plot(histo_bins[1:], gt_histo, label='gt')
    plt.plot(histo_bins[1:], exp_histo, label='exp')
    plt.savefig('./logs_%s/%s_final_histo.png' % (dataset, prefix))
    plt.close()
    gt_min, gt_max, exp_min, exp_max, l1, l2, l1_01, l2_01 = get_limits(
        dataloader, bm_fct, [gt_scale, gt_clamp_trans],
        [exp_scale, exp_histo_specification])
    print('GT  min: %8.4f  --  max: %8.4f' % (gt_min, gt_max))
    print('EXP min: %8.4f  --  max: %8.4f' % (exp_min, exp_max))
    print('L1 : %8.4f' % (l1))
    print('L2 : %8.4f' % (l2))
    print('L1 [0,1] : %8.4f' % (l1_01))
    print('L2 [0,1] : %8.4f' % (l2_01))
    # save example
    save_path = '%s_example.png' % (prefix)
    save_example(dataloader, bm_fct, save_path, [gt_scale, gt_clamp_trans],
                 [exp_scale, exp_histo_specification])


def run_analysis_v2(dataset, split_name, exp_name, splits, db_type, gt_path,
                    rgb_path, bm_fct, gt_th1, gt_th2):
    n_bins_gen_histo = 1000
    n_bins_histospecification = 100000
    prefix = 'v2_%s_%s_%s_%04.2f_%04.2f' % (dataset, split_name, exp_name,
                                            gt_th1, gt_th2)
    print('%s' % (prefix))
    db = db_type(gt_path, rgb_path, splits)
    dataloader = DataLoader(db, batch_size=30, shuffle=False, num_workers=20)
    # Step 0 - get ranges for both gt and expert results & errors
    gt_min, gt_max, exp_min, exp_max, l1, l2, l1_01, l2_01 = get_limits(
        dataloader, bm_fct, [], [])
    print('GT  min: %8.4f  --  max: %8.4f' % (gt_min, gt_max))
    print('EXP min: %8.4f  --  max: %8.4f' % (exp_min, exp_max))
    print('L1 : %8.4f' % (l1))
    print('L2 : %8.4f' % (l2))
    print('L1 [0,1] : %8.4f' % (l1_01))
    print('L2 [0,1] : %8.4f' % (l2_01))

    # Step 1 - scale all in range [0,1]
    gt_scale = TransFct_ScaleMinMax(gt_min, gt_max)
    exp_scale = TransFct_ScaleMinMax(exp_min, exp_max)
    np.save('%s_gt_min.npy' % (prefix), gt_min)
    np.save('%s_gt_max.npy' % (prefix), gt_max)
    np.save('%s_exp_min.npy' % (prefix), exp_min)
    np.save('%s_exp_max.npy' % (prefix), exp_max)

    gt_min, gt_max, exp_min, exp_max, l1, l2, l1_01, l2_01 = get_limits(
        dataloader, bm_fct, [gt_scale], [exp_scale])
    print('GT  min: %8.4f  --  max: %8.4f' % (gt_min, gt_max))
    print('EXP min: %8.4f  --  max: %8.4f' % (exp_min, exp_max))
    print('L1 : %8.4f' % (l1))
    print('L2 : %8.4f' % (l2))
    print('L1 [0,1] : %8.4f' % (l1_01))
    print('L2 [0,1] : %8.4f' % (l2_01))

    # Step 2 - compute histograms of scaled values
    csv_path = './logs_%s/%s_initial_histo.csv' % (dataset, prefix)
    suffix = '%s_all' % (prefix)
    histo_bins, gt_histo, cum_gt_histo, exp_histo, cum_exp_histo = get_histogram(
        dataloader, bm_fct, n_bins_gen_histo, csv_path, suffix, [gt_scale],
        [exp_scale], 0, 1)
    # save plot
    plt.plot(histo_bins[1:], gt_histo, label='gt')
    plt.plot(histo_bins[1:], exp_histo, label='exp')
    plt.legend()
    plt.savefig('./logs_%s/%s_initial_histo.png' % (dataset, prefix))
    plt.close()

    # get gt th for gt_th1 of histogram and gt_th2

    pos = np.argwhere(cum_gt_histo >= gt_th1)[0]
    gt_th1_value = histo_bins[pos]
    pos = np.argwhere(cum_gt_histo >= gt_th2)[0]
    gt_th2_value = histo_bins[pos]
    np.save('%s_gt_th1.npy' % prefix, gt_th1_value)
    np.save('%s_gt_th2.npy' % prefix, gt_th2_value)
    gt_clamp_trans = TransFct_HistoClamp(gt_th1_value, gt_th2_value)

    pos = np.argwhere(cum_exp_histo >= gt_th1)[0]
    exp_th1_value = histo_bins[pos]
    pos = np.argwhere(cum_exp_histo >= gt_th2)[0]
    exp_th2_value = histo_bins[pos]
    np.save('%s_exp_th1.npy' % prefix, exp_th1_value)
    np.save('%s_exp_th2.npy' % prefix, exp_th2_value)
    exp_clamp_trans = TransFct_HistoClamp(exp_th1_value, exp_th2_value)

    # Step 3 - get limits & eval for scaled and clamped gt (95%)
    gt_min, gt_max, exp_min, exp_max, l1, l2, l1_01, l2_01 = get_limits(
        dataloader, bm_fct, [gt_scale, gt_clamp_trans],
        [exp_scale, exp_clamp_trans])
    print('GT  min: %8.4f  --  max: %8.4f' % (gt_min, gt_max))
    print('EXP min: %8.4f  --  max: %8.4f' % (exp_min, exp_max))
    print('L1 : %8.4f' % (l1))
    print('L2 : %8.4f' % (l2))
    print('L1 [0,1] : %8.4f' % (l1_01))
    print('L2 [0,1] : %8.4f' % (l2_01))

    csv_path = './logs_%s/%s_afterScale_and_GTClamp_histo.csv' % (dataset,
                                                                  prefix)
    suffix = '%s_all' % (prefix)
    histo_bins, gt_histo, cum_gt_histo, exp_histo, cum_exp_histo = get_histogram(
        dataloader, bm_fct, n_bins_gen_histo, csv_path, suffix,
        [gt_scale, gt_clamp_trans], [exp_scale, exp_clamp_trans], 0, 1)
    # save plot
    plt.plot(histo_bins[1:], gt_histo, label='gt')
    plt.plot(histo_bins[1:], exp_histo, label='exp')
    plt.legend()
    plt.savefig('./logs_%s/%s_GTkeep95p_histo.png' % (dataset, prefix))
    plt.close()

    # save example
    save_path = '%s_example.png' % (prefix)
    save_example(dataloader, bm_fct, save_path, [gt_scale, gt_clamp_trans],
                 [exp_scale, exp_clamp_trans])


def run_analysis_v3(dataset, split_name, exp_name, splits, db_type, gt_path,
                    rgb_path, bm_fct, gt_th1, gt_th2):
    n_bins_gen_histo = 1000
    n_bins_histospecification = 100000
    prefix = 'v3_%s_%s_%s_%04.2f_%04.2f' % (dataset, split_name, exp_name,
                                            gt_th1, gt_th2)
    print('%s' % (prefix))
    db = db_type(gt_path, rgb_path, splits)
    dataloader = DataLoader(db, batch_size=30, shuffle=False, num_workers=20)
    # Step 0 - get ranges for both gt and expert results & errors
    gt_min, gt_max, exp_min, exp_max, l1, l2, l1_01, l2_01 = get_limits(
        dataloader, bm_fct, [], [])
    print('GT  min: %8.4f  --  max: %8.4f' % (gt_min, gt_max))
    print('EXP min: %8.4f  --  max: %8.4f' % (exp_min, exp_max))
    print('L1 : %8.4f' % (l1))
    print('L2 : %8.4f' % (l2))
    print('L1 [0,1] : %8.4f' % (l1_01))
    print('L2 [0,1] : %8.4f' % (l2_01))

    # Step 1 - scale all in range [0,1]
    gt_scale = TransFct_ScaleMinMax(gt_min, gt_max)
    exp_scale = TransFct_ScaleMinMax(exp_min, exp_max)
    np.save('%s_gt_min.npy' % (prefix), gt_min)
    np.save('%s_gt_max.npy' % (prefix), gt_max)
    np.save('%s_exp_min.npy' % (prefix), exp_min)
    np.save('%s_exp_max.npy' % (prefix), exp_max)

    gt_min, gt_max, exp_min, exp_max, l1, l2, l1_01, l2_01 = get_limits(
        dataloader, bm_fct, [gt_scale], [exp_scale])
    print('GT  min: %8.4f  --  max: %8.4f' % (gt_min, gt_max))
    print('EXP min: %8.4f  --  max: %8.4f' % (exp_min, exp_max))
    print('L1 : %8.4f' % (l1))
    print('L2 : %8.4f' % (l2))
    print('L1 [0,1] : %8.4f' % (l1_01))
    print('L2 [0,1] : %8.4f' % (l2_01))

    # Step 2 - compute histograms of scaled values
    csv_path = './logs_%s/%s_initial_histo.csv' % (dataset, prefix)
    suffix = '%s_all' % (prefix)
    histo_bins, gt_histo, cum_gt_histo, exp_histo, cum_exp_histo = get_histogram(
        dataloader, bm_fct, n_bins_gen_histo, csv_path, suffix, [gt_scale],
        [exp_scale], 0, 1)
    # save plot
    plt.plot(histo_bins[1:], gt_histo, label='gt')
    plt.plot(histo_bins[1:], exp_histo, label='exp')
    plt.legend()
    plt.savefig('./logs_%s/%s_initial_histo.png' % (dataset, prefix))
    plt.close()

    gt_histo_specification = TransFct_HistoSpecification_GT(
        dataloader, bm_fct, [gt_scale], [exp_scale], n_bins_histospecification)

    # Step 3 - get limits & eval for transformed gt_histo

    gt_min, gt_max, exp_min, exp_max, l1, l2, l1_01, l2_01 = get_limits(
        dataloader, bm_fct, [gt_scale, gt_histo_specification], [exp_scale])
    print('GT  min: %8.4f  --  max: %8.4f' % (gt_min, gt_max))
    print('EXP min: %8.4f  --  max: %8.4f' % (exp_min, exp_max))
    print('L1 : %8.4f' % (l1))
    print('L2 : %8.4f' % (l2))
    print('L1 [0,1] : %8.4f' % (l1_01))
    print('L2 [0,1] : %8.4f' % (l2_01))

    csv_path = './logs_%s/%s_afterScale_and_gt_trans_histo.csv' % (dataset,
                                                                   prefix)
    suffix = '%s_all' % (prefix)
    histo_bins, gt_histo, cum_gt_histo, exp_histo, cum_exp_histo = get_histogram(
        dataloader, bm_fct, n_bins_gen_histo, csv_path, suffix,
        [gt_scale, gt_histo_specification], [exp_scale], 0, 1)
    # save plot
    plt.plot(histo_bins[1:], gt_histo, label='gt')
    plt.plot(histo_bins[1:], exp_histo, label='exp')
    plt.legend()
    plt.savefig('./logs_%s/%s_afterScale_and_gt_trans_histo.png' %
                (dataset, prefix))
    plt.close()

    exp_histo_specification = TransFct_HistoSpecification_Exp(
        dataloader, bm_fct, [gt_scale, gt_histo_specification], [exp_scale],
        n_bins_histospecification)
    np.save('%s_n_bins.npy' % prefix, exp_histo_specification.n_bins)
    np.save('%s_cum_exp_histo.npy' % prefix,
            exp_histo_specification.cum_data_histo)
    np.save('%s_inv_cum_gt_histo.npy' % prefix,
            exp_histo_specification.inv_cum_target_histo)
    gt_min, gt_max, exp_min, exp_max, l1, l2, l1_01, l2_01 = get_limits(
        dataloader, bm_fct, [gt_scale, gt_histo_specification],
        [exp_scale, exp_histo_specification])
    print('GT  min: %8.4f  --  max: %8.4f' % (gt_min, gt_max))
    print('EXP min: %8.4f  --  max: %8.4f' % (exp_min, exp_max))
    print('L1 : %8.4f' % (l1))
    print('L2 : %8.4f' % (l2))
    print('L1 [0,1] : %8.4f' % (l1_01))
    print('L2 [0,1] : %8.4f' % (l2_01))

    csv_path = './logs_%s/%s_final_histo.csv' % (dataset, prefix)
    suffix = '%s_all' % (prefix)
    histo_bins, gt_histo, cum_gt_histo, exp_histo, cum_exp_histo = get_histogram(
        dataloader, bm_fct, n_bins_gen_histo, csv_path, suffix,
        [gt_scale, gt_histo_specification],
        [exp_scale, exp_histo_specification], 0, 1)
    # save plot
    plt.plot(histo_bins[1:], gt_histo, label='gt')
    plt.plot(histo_bins[1:], exp_histo, label='exp')
    plt.savefig('./logs_%s/%s_final_histo.png' % (dataset, prefix))
    plt.close()
    gt_min, gt_max, exp_min, exp_max, l1, l2, l1_01, l2_01 = get_limits(
        dataloader, bm_fct, [gt_scale, gt_histo_specification],
        [exp_scale, exp_histo_specification])
    print('GT  min: %8.4f  --  max: %8.4f' % (gt_min, gt_max))
    print('EXP min: %8.4f  --  max: %8.4f' % (exp_min, exp_max))
    print('L1 : %8.4f' % (l1))
    print('L2 : %8.4f' % (l2))
    print('L1 [0,1] : %8.4f' % (l1_01))
    print('L2 [0,1] : %8.4f' % (l2_01))
    # save example
    save_path = '%s_example.png' % (prefix)
    save_example(dataloader, bm_fct, save_path,
                 [gt_scale, gt_histo_specification],
                 [exp_scale, exp_histo_specification])


def run_analysis_v4(dataset, split_name, exp_name, splits, db_type, gt_path,
                    rgb_path, bm_fct):

    n_bins_gen_histo = 1000
    n_bins_histospecification = 100000
    prefix = 'v4_%s_%s_%s' % (dataset, split_name, exp_name),

    print('%s' % (prefix))
    db = db_type(gt_path, rgb_path, splits)
    dataloader = DataLoader(db, batch_size=30, shuffle=False, num_workers=20)
    # Step 0 - get ranges for both gt and expert results & errors
    gt_min, gt_max, exp_min, exp_max, l1, l2, l1_01, l2_01 = get_limits(
        dataloader, bm_fct, [], [])
    print('GT  min: %8.4f  --  max: %8.4f' % (gt_min, gt_max))
    print('EXP min: %8.4f  --  max: %8.4f' % (exp_min, exp_max))
    print('L1 : %8.4f' % (l1))
    print('L2 : %8.4f' % (l2))
    print('L1 [0,1] : %8.4f' % (l1_01))
    print('L2 [0,1] : %8.4f' % (l2_01))

    # Step 1 - scale all in range [0,1]
    gt_scale = TransFct_ScaleMinMax(gt_min, gt_max)
    exp_scale = TransFct_ScaleMinMax(exp_min, exp_max)
    np.save('%s_gt_min.npy' % (prefix), gt_min)
    np.save('%s_gt_max.npy' % (prefix), gt_max)
    np.save('%s_exp_min.npy' % (prefix), exp_min)
    np.save('%s_exp_max.npy' % (prefix), exp_max)

    gt_min, gt_max, exp_min, exp_max, l1, l2, l1_01, l2_01 = get_limits(
        dataloader, bm_fct, [gt_scale], [exp_scale])
    print('GT  min: %8.4f  --  max: %8.4f' % (gt_min, gt_max))
    print('EXP min: %8.4f  --  max: %8.4f' % (exp_min, exp_max))
    print('L1 : %8.4f' % (l1))
    print('L2 : %8.4f' % (l2))
    print('L1 [0,1] : %8.4f' % (l1_01))
    print('L2 [0,1] : %8.4f' % (l2_01))

    # Step 2 - compute histograms of scaled values
    csv_path = './logs_%s/%s_initial_histo.csv' % (dataset, prefix)
    suffix = '%s_all' % (prefix)
    histo_bins, gt_histo, cum_gt_histo, exp_histo, cum_exp_histo = get_histogram(
        dataloader, bm_fct, n_bins_gen_histo, csv_path, suffix, [gt_scale],
        [exp_scale], 0, 1)
    # save plot
    plt.plot(histo_bins[1:], gt_histo, label='gt')
    plt.plot(histo_bins[1:], exp_histo, label='exp')
    plt.legend()
    plt.savefig('./logs_%s/%s_initial_histo.png' % (dataset, prefix))
    plt.close()

    gt_histo_specification = TransFct_HistoSpecification_GT(
        dataloader, bm_fct, [gt_scale], [exp_scale], n_bins_histospecification)
    exp_histo_specification = TransFct_HistoSpecification_Exp_v2(
        dataloader, bm_fct, [gt_scale], [exp_scale], n_bins_histospecification)

    np.save('%s_gt_n_bins.npy' % prefix, gt_histo_specification.n_bins)
    np.save('%s_gt_cum_data_histo.npy' % prefix,
            gt_histo_specification.cum_data_histo)
    np.save('%s_gt_inv_cum_target_histo.npy' % prefix,
            gt_histo_specification.inv_cum_target_histo)

    np.save('%s_exp_n_bins.npy' % prefix, exp_histo_specification.n_bins)
    np.save('%s_exp_cum_data_histo.npy' % prefix,
            exp_histo_specification.cum_data_histo)
    np.save('%s_exp_inv_cum_target_histo.npy' % prefix,
            exp_histo_specification.inv_cum_target_histo)

    # Step 3 - get limits & eval for transformed gt_histo
    gt_min, gt_max, exp_min, exp_max, l1, l2, l1_01, l2_01 = get_limits(
        dataloader, bm_fct, [gt_scale, gt_histo_specification],
        [exp_scale, exp_histo_specification])
    print('GT  min: %8.4f  --  max: %8.4f' % (gt_min, gt_max))
    print('EXP min: %8.4f  --  max: %8.4f' % (exp_min, exp_max))
    print('L1 : %8.4f' % (l1))
    print('L2 : %8.4f' % (l2))
    print('L1 [0,1] : %8.4f' % (l1_01))
    print('L2 [0,1] : %8.4f' % (l2_01))

    csv_path = './logs_%s/%s_after_scale_and_trans_histo.csv' % (dataset,
                                                                 prefix)
    suffix = '%s_all' % (prefix)
    histo_bins, gt_histo, cum_gt_histo, exp_histo, cum_exp_histo = get_histogram(
        dataloader, bm_fct, n_bins_gen_histo, csv_path, suffix,
        [gt_scale, gt_histo_specification],
        [exp_scale, exp_histo_specification], 0, 1)
    # save plot
    plt.plot(histo_bins[1:], gt_histo, label='gt')
    plt.plot(histo_bins[1:], exp_histo, label='exp')
    plt.legend()
    plt.savefig('./logs_%s/%s_final_histo.png' % (dataset, prefix))
    plt.close()

    # save example
    save_path = '%s_example.png' % (prefix)
    save_example(dataloader, bm_fct, save_path,
                 [gt_scale, gt_histo_specification],
                 [exp_scale, exp_histo_specification])


if __name__ == "__main__":
    argv = sys.argv
    exp_name = argv[1]
    dataset = argv[2]
    gt_th1 = float(argv[3])
    gt_th2 = float(argv[4])
    splits = argv[5:]

    assert (exp_name == 'xtc' or exp_name == 'sgdepth')
    if exp_name == 'xtc':
        depth_expert = experts.depth_expert.DepthModelXTC(full_expert=True)
    else:
        sys.argv = ['']
        depth_expert = experts.depth_expert.DepthModel(full_expert=True)

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
    if dataset == 'replica2':
        db_type = Replica_RGB_and_Depth_DB
        gt_path = replica2_gt_path
        rgb_path = replica2_rgb_path
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

    if argv[5] == 'all':
        run_analysis_v4(dataset, 'all', exp_name, splits, db_type, gt_path,
                        rgb_path, bm_fct)
        '''
        print(
            '===================================================================='
        )
        run_analysis_v3(dataset, 'all', exp_name, splits, db_type, gt_path,
                        rgb_path, bm_fct, gt_th1, gt_th2)
        print(
            '===================================================================='
        )
        run_analysis(dataset, 'all', exp_name, splits, db_type, gt_path,
                     rgb_path, bm_fct, gt_th1, gt_th2)
        print(
            '===================================================================='
        )
        run_analysis_v2(dataset, 'all', exp_name, splits, db_type, gt_path,
                        rgb_path, bm_fct, gt_th1, gt_th2)
        print(
            '===================================================================='
        )
        '''
    else:
        for split_name in splits:
            run_analysis(dataset, split_name, exp_name, [split_name], db_type,
                         gt_path, rgb_path, bm_fct, gt_th1, gt_th2)
