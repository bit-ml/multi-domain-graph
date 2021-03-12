import glob
import os
import pathlib
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
'''
conf => take default value from config 
STORE => take values from iter STORE_PATH
--------------------------------------------------------------------------------------------------
                                       config 1                config 2
--------------------------------------------------------------------------------------------------
iter_idx == 1 |  train - src |          conf            |       conf
                 train - dst |          conf            |       conf
                 ---------------------------------------------------------------------------------
                 valid - src |          conf            |       conf
                 valid - dst |          conf            |       conf
                 ---------------------------------------------------------------------------------
                 test - src  |          conf            |       conf
                 test - dst  |          conf            |       conf
                 test - gt   |          conf            |       conf
--------------------------------------------------------------------------------------------------
iter_idx > 1  |  train - src |          conf            |       STORE
                 train - dst |          STORE           |       STORE
                 ---------------------------------------------------------------------------------
                 valid - src |          conf            |       STORE
                 valid - dst |          STORE           |       STORE
                 ---------------------------------------------------------------------------------
                 test - src  |          conf            |       STORE
                 test - dst  |          STORE           |       STORE
                 test - gt   |          conf            |       conf
--------------------------------------------------------------------------------------------------                
'''

experts_using_gt_in_second_iter = ['rgb', 'hsv', 'grayscale', 'halftone_gray']


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


def get_paths_for_idx_and_split(config, iter_idx, split_str, running_iter_idx,
                                for_next_iter_idx_subset, src_expert,
                                dst_expert):
    iters_config = config.getint('General', 'iters_config')

    if running_iter_idx == 1:
        src_path = config.get('PathsIter%d' % iter_idx, 'ITER%d_%s_SRC_PATH' %
                              (iter_idx, split_str)).split('\n')
        dst_path = config.get('PathsIter%d' % iter_idx, 'ITER%d_%s_DST_PATH' %
                              (iter_idx, split_str)).split('\n')
    else:
        if (iters_config == 1) or (src_expert.identifier
                                   in experts_using_gt_in_second_iter):
            src_path = config.get('PathsIter%d' % iter_idx,
                                  'ITER%d_%s_SRC_PATH' %
                                  (iter_idx, split_str)).split('\n')
        else:  # iters_config == 2
            src_path = config.get(
                'PathsIter%d' % iter_idx,
                'ITER%d_%s_STORE_PATH' % (iter_idx, split_str)).split('\n')
        if dst_expert.identifier in experts_using_gt_in_second_iter:
            dst_path = config.get('PathsIter%d' % iter_idx,
                                  'ITER%d_%s_DST_PATH' %
                                  (iter_idx, split_str)).split('\n')
        else:
            dst_path = config.get(
                'PathsIter%d' % iter_idx,
                'ITER%d_%s_STORE_PATH' % (iter_idx, split_str)).split('\n')

    patterns = config.get('PathsIter%d' % iter_idx, 'ITER%d_%s_PATTERNS' %
                          (iter_idx, split_str)).split('\n')
    first_k = config.getint('PathsIter%d' % iter_idx,
                            'ITER%d_%s_FIRST_K' % (iter_idx, split_str))
    if split_str == 'TEST':
        gt_dst_path = config.get(
            'PathsIter%d' % iter_idx,
            'ITER%d_%s_GT_DST_PATH' % (iter_idx, split_str)).split('\n')
    else:
        gt_dst_path = None

    # If we get paths for next iter => we will only get a certain subset of data(e.g. part1 or part2 for training set)
    if not (running_iter_idx == iter_idx):
        if len(patterns) == len(src_path):
            patterns = patterns[
                for_next_iter_idx_subset:for_next_iter_idx_subset + 1]
        src_path = src_path[for_next_iter_idx_subset:for_next_iter_idx_subset +
                            1]
        dst_path = dst_path[for_next_iter_idx_subset:for_next_iter_idx_subset +
                            1]
        if not (gt_dst_path == None):
            gt_dst_path = gt_dst_path[
                for_next_iter_idx_subset:for_next_iter_idx_subset + 1]

    return src_path, dst_path, patterns, first_k, gt_dst_path


def get_glob_paths(path, identifier, patterns):
    glob_paths = []
    for i in range(len(path)):

        if len(patterns) == len(path):
            glob_paths_ = ["%s/%s/%s.npy" % (path[i], identifier, patterns[i])]
        elif len(patterns) == 1:
            glob_paths_ = ["%s/%s/%s.npy" % (path[i], identifier, patterns[0])]
        else:
            glob_paths_ = [
                "%s/%s/%s.npy" % (path[i], identifier, pattern)
                for pattern in patterns
            ]
        glob_paths = glob_paths + glob_paths_
    return glob_paths


class ImageLevelDataset(Dataset):
    def __init__(self,
                 src_expert,
                 dst_expert,
                 config,
                 iter_idx,
                 split_str,
                 for_next_iter=False,
                 for_next_iter_idx_subset=0):
        """
            src_expert
            dst_expert 
            config 
            iter_idx - current iteration index 
            split_str - desired split ('TRAIN', 'VALID' or 'TEST')
            for_next_iter - if we load a dataset for next iter (load it in order to save ensembles of current iter)
            for_next_iter_idx_subset - if we load a dataset for next iter, we can load per subset s.t. we can save it in corresponding subset dst 
        """
        super(ImageLevelDataset, self).__init__()
        self.src_expert = src_expert
        self.dst_expert = dst_expert

        src_path, dst_path, patterns, first_k, gt_dst_path = get_paths_for_idx_and_split(
            config, iter_idx, split_str,
            (iter_idx - 1) if for_next_iter else iter_idx,
            for_next_iter_idx_subset, src_expert, dst_expert)

        paths_str = [
            pathlib.Path(src_path_).parts[-1] for src_path_ in src_path
        ]
        paths_str = '_'.join(paths_str)

        tag = 'iter_%d_split_%s_nPaths_%d_%s_%s_iters_config_%d_%d' % (
            iter_idx, split_str, len(src_path), paths_str,
            '_for_next_iter' if for_next_iter else '',
            config.getint('General', 'iters_config'), for_next_iter_idx_subset)

        CACHE_NAME = config.get('General', 'CACHE_NAME')

        if not for_next_iter and gt_dst_path is not None:
            for i in range(len(gt_dst_path)):
                available_gt_domains = os.listdir(gt_dst_path[i])
                if self.dst_expert.domain_name not in available_gt_domains:
                    self.src_paths = []
                    self.dst_paths = []
                    self.gt_dst_paths = []
                    return

        if first_k == 0:
            self.src_paths = []
            self.dst_paths = []
            self.gt_dst_paths = []
            return

        #s = time.time()
        cache_src = "%s/src_%s_%s.npy" % (CACHE_NAME, tag,
                                          self.src_expert.identifier)
        glob_paths_srcs = get_glob_paths(src_path, self.src_expert.identifier,
                                         patterns)
        self.src_paths = load_glob_with_cache_multiple_patterns(
            cache_src, glob_paths_srcs)
        self.src_paths = self.src_paths[:len(self.src_paths) if first_k ==
                                        -1 else min(first_k, len(self.src_paths
                                                                 ))]

        #print("Load %s %20.10f" % (cache_src, time.time() - s))

        #s = time.time()
        cache_dst = "%s/dst_%s_%s.npy" % (CACHE_NAME, tag,
                                          self.dst_expert.identifier)
        glob_paths_dsts = get_glob_paths(dst_path, self.dst_expert.identifier,
                                         patterns)
        self.dst_paths = load_glob_with_cache_multiple_patterns(
            cache_dst, glob_paths_dsts)
        self.dst_paths = self.dst_paths[:len(self.dst_paths) if first_k ==
                                        -1 else min(first_k, len(self.dst_paths
                                                                 ))]

        #print("Load %s %20.10f" % (cache_dst, time.time() - s))

        if (not for_next_iter) and (not (gt_dst_path == None)):
            #s = time.time()
            cache_gt_dst = "%s/gt_dst_%s_%s.npy" % (
                CACHE_NAME, tag, self.dst_expert.domain_name)
            glob_paths_gt_dsts = get_glob_paths(gt_dst_path,
                                                self.dst_expert.domain_name,
                                                patterns)
            self.gt_dst_paths = load_glob_with_cache_multiple_patterns(
                cache_gt_dst, glob_paths_gt_dsts)
            self.gt_dst_paths = self.gt_dst_paths[:len(
                self.gt_dst_paths
            ) if first_k == -1 else min(first_k, len(self.gt_dst_paths))]

            #print("Load %s %20.10f" % (cache_gt_dst, time.time() - s))
        else:
            self.gt_dst_paths = []

        if not (first_k == 0):
            assert (len(self.src_paths) == len(self.dst_paths))
            if (not for_next_iter) and (not (gt_dst_path == None)):
                assert (len(self.src_paths) == len(self.gt_dst_paths))

    def __getitem__(self, index):
        src_data = np.load(self.src_paths[index]).astype(np.float32)
        dst_data = np.load(self.dst_paths[index])

        if len(self.gt_dst_paths) > 0:
            gt_dst_data = np.load(self.gt_dst_paths[index])
            return src_data, dst_data, gt_dst_data
        else:
            return src_data, dst_data

    def __len__(self):
        return len(self.src_paths)
