import glob
import os
import pathlib
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# TODO change s.t. -1 means all samples
first_k = 3000
first_k_test = 9464  #60#64
CACHE_NAME = "my_cache"
W, H = 256, 256


def load_glob_with_cache(cache_file, glob_path):
    if not os.path.exists(cache_file):
        all_paths = sorted(glob.glob(glob_path))
        save_folder = os.path.dirname(cache_file)
        if not os.path.exists(save_folder):
            pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)
        np.save(cache_file, all_paths)
    else:
        all_paths = np.load(cache_file)
    return all_paths


class Domain2DDataset(Dataset):
    def __init__(self, rgbs_path, experts_path, dataset_path, experts):
        super(Domain2DDataset, self).__init__()
        self.experts = experts

        pattern = "/*/*00001"
        s = time.time()

        tag = pathlib.Path(dataset_path).parts[-1]
        # load all rgbs paths
        cache_rgb = "%s/rgbs_paths_%s_%s.npy" % (CACHE_NAME, tag, pattern[-3:])
        glob_path_rgb = "%s/%s/%s.jpg" % (rgbs_path, dataset_path, pattern)
        self.rgb_paths = load_glob_with_cache(cache_rgb,
                                              glob_path_rgb)[:first_k]

        # load experts paths
        cache_e1 = "%s/%s_%s_%s.npy" % (CACHE_NAME, self.experts[0].identifier,
                                        tag, pattern[-3:])
        glob_path_e1 = "%s/%s/%s/%s.npy" % (
            experts_path, dataset_path, self.experts[0].identifier, pattern)
        self.e1_output_path = load_glob_with_cache(cache_e1,
                                                   glob_path_e1)[:first_k]

        cache_e2 = "%s/%s_%s_%s.npy" % (CACHE_NAME, self.experts[1].identifier,
                                        tag, pattern[-3:])
        glob_path_e2 = "%s/%s/%s/%s.npy" % (
            experts_path, dataset_path, self.experts[1].identifier, pattern)
        self.e2_output_path = load_glob_with_cache(cache_e2,
                                                   glob_path_e2)[:first_k]
        e = time.time()

        assert (len(self.rgb_paths) == len(self.e1_output_path) == len(
            self.e2_output_path))

        # TODO: precompute+save mean & std when buliding cache

    def __getitem__(self, index):
        oe1 = np.load(self.e1_output_path[index])
        oe2 = np.load(self.e2_output_path[index])
        return oe1, oe2

    def __len__(self):
        return len(self.rgb_paths)


class DomainTestDataset(Dataset):
    def __init__(self, preproc_gt_path, experts_path, dataset_path, experts):
        super(DomainTestDataset, self).__init__()
        self.experts = experts

        available_experts = os.listdir(os.path.join(experts_path,
                                                    dataset_path))
        available_gts = os.listdir(os.path.join(preproc_gt_path, dataset_path))
        if self.experts[0].identifier in available_experts and \
            self.experts[1].identifier in available_experts and \
            self.experts[1].domain_name in available_gts:
            self.available = True
        else:
            self.available = False
            return

        pattern = "*"

        # get data for src expert
        cache_e1 = "%s/test_%s_pseudo_gt.npy" % (CACHE_NAME,
                                                 self.experts[0].identifier)
        glob_path_e1 = "%s/%s/%s/%s.npy" % (
            experts_path, dataset_path, self.experts[0].identifier, pattern)
        self.e1_output_path = load_glob_with_cache(cache_e1,
                                                   glob_path_e1)[:first_k_test]

        # get data for dst expert
        cache_e2 = "%s/test_%s_pseudo_gt.npy" % (CACHE_NAME,
                                                 self.experts[1].identifier)
        glob_path_e2 = "%s/%s/%s/%s.npy" % (
            experts_path, dataset_path, self.experts[1].identifier, pattern)
        self.e2_output_path = load_glob_with_cache(cache_e2,
                                                   glob_path_e2)[:first_k_test]

        # get data for domain of dst expert
        cache_d2_gt = "%s/test_%s_gt.npy" % (CACHE_NAME,
                                             self.experts[1].domain_name)
        glob_path_d2_gt = "%s/%s/%s/%s.npy" % (preproc_gt_path, dataset_path,
                                               self.experts[1].domain_name,
                                               pattern)
        self.d2_gt_output_path = load_glob_with_cache(
            cache_d2_gt, glob_path_d2_gt)[:first_k_test]

        # check data
        assert (len(self.e1_output_path) == len(self.e2_output_path) == len(
            self.d2_gt_output_path))

    def __getitem__(self, index):
        if self.available == False:
            return None, None, None
        oe1 = np.load(self.e1_output_path[index])
        oe2 = np.load(self.e2_output_path[index])
        d2_gt = np.load(self.d2_gt_output_path[index])
        return oe1, oe2, d2_gt

    def __len__(self):
        if self.available == False:
            return 0
        return len(self.e1_output_path)
