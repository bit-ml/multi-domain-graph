import glob
import os
import pathlib
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CACHE_NAME = "my_cache"
W, H = 256, 256


def load_glob_with_cache_multiple_patterns(cache_file, glob_paths):
    if not os.path.exists(cache_file):
        all_paths = sorted(glob.glob(glob_paths[0]))
        for i in np.arange(1, len(glob_paths)):
            all_paths = all_paths + glob.glob(glob_paths[i])
        all_paths = sorted(all_paths)

        save_folder = os.path.dirname(cache_file)
        if not os.path.exists(save_folder):
            pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)
        np.save(cache_file, all_paths)
    else:
        all_paths = np.load(cache_file)
    return all_paths


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
    def __init__(self, dataset_path, experts, patterns, first_k):
        super(Domain2DDataset, self).__init__()
        self.experts = experts

        s = time.time()
        tag = pathlib.Path(dataset_path).parts[-1]

        # load experts paths
        cache_e1 = "%s/%s_%s_%d.npy" % (
            CACHE_NAME, tag, self.experts[0].identifier, len(patterns))
        glob_paths_e1 = [
            "%s/%s/%s.npy" %
            (dataset_path, self.experts[0].identifier, pattern)
            for pattern in patterns
        ]

        self.e1_output_path = load_glob_with_cache_multiple_patterns(
            cache_e1, glob_paths_e1)
        self.e1_output_path = self.e1_output_path[:len(self.e1_output_path
                                                       ) if first_k ==
                                                  -1 else first_k]

        cache_e2 = "%s/%s_%s_%d.npy" % (
            CACHE_NAME, tag, self.experts[1].identifier, len(patterns))
        glob_paths_e2 = [
            "%s/%s/%s.npy" %
            (dataset_path, self.experts[1].identifier, pattern)
            for pattern in patterns
        ]
        self.e2_output_path = load_glob_with_cache_multiple_patterns(
            cache_e2, glob_paths_e2)
        self.e2_output_path = self.e2_output_path[:len(self.e2_output_path
                                                       ) if first_k ==
                                                  -1 else first_k]
        e = time.time()

        assert (len(self.e1_output_path) == len(self.e2_output_path))

        # TODO: precompute+save mean & std when buliding cache

    def __getitem__(self, index):
        oe1 = np.load(self.e1_output_path[index])
        oe2 = np.load(self.e2_output_path[index])
        return oe1, oe2

    def __len__(self):
        return len(self.e1_output_path)


class DomainTestDataset(Dataset):
    def __init__(self, preproc_gt_path, experts_path, dataset_path, experts,
                 first_k):
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
                                                   glob_path_e1)[:first_k]

        # get data for dst expert
        cache_e2 = "%s/test_%s_pseudo_gt.npy" % (CACHE_NAME,
                                                 self.experts[1].identifier)
        glob_path_e2 = "%s/%s/%s/%s.npy" % (
            experts_path, dataset_path, self.experts[1].identifier, pattern)
        self.e2_output_path = load_glob_with_cache(cache_e2,
                                                   glob_path_e2)[:first_k]

        # get data for domain of dst expert
        cache_d2_gt = "%s/test_%s_gt.npy" % (CACHE_NAME,
                                             self.experts[1].domain_name)
        glob_path_d2_gt = "%s/%s/%s/%s.npy" % (preproc_gt_path, dataset_path,
                                               self.experts[1].domain_name,
                                               pattern)
        self.d2_gt_output_path = load_glob_with_cache(
            cache_d2_gt, glob_path_d2_gt)[:first_k]

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
