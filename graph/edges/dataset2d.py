import glob
import os
import pathlib
import time

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

first_k = 3000

def load_with_cache(cache_file, glob_path):
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
        cache_rgb = "my_cache/rgbs_paths_%s_%s.npy" % (tag, pattern[-3:])
        glob_path_rgb = "%s/%s/%s.jpg" % (rgbs_path, dataset_path, pattern)
        self.rgb_paths = load_with_cache(cache_rgb, glob_path_rgb)[:first_k]

        # load experts paths
        cache_e1 = "my_cache/%s_%s_%s.npy" % (self.experts[0].str_id, tag,
                                              pattern[-3:])
        glob_path_e1 = "%s/%s/%s/%s.npy" % (
            experts_path, self.experts[0].str_id, dataset_path, pattern)
        self.e1_output_path = load_with_cache(cache_e1, glob_path_e1)[:first_k]

        cache_e2 = "my_cache/%s_%s_%s.npy" % (self.experts[1].str_id, tag,
                                              pattern[-3:])
        glob_path_e2 = "%s/%s/%s/%s.npy" % (
            experts_path, self.experts[1].str_id, dataset_path, pattern)
        self.e2_output_path = load_with_cache(cache_e2, glob_path_e2)[:first_k]
        e = time.time()

        # print("glob time:", e - s)
        print("Dataset size", len(self.rgb_paths), len(self.e1_output_path),
              len(self.e2_output_path))
        assert (len(self.rgb_paths) == len(self.e1_output_path) == len(
            self.e2_output_path))
        # print(self.rgb_paths[0])
        # print(self.e1_output_path[0])
        # print(self.e2_output_path[0])

        # TODO: precompute+save mean & std when buliding cache

    def __getitem__(self, index):
        oe1 = np.load(self.e1_output_path[index])
        oe2 = np.load(self.e2_output_path[index])
        return oe1, oe2

    def __len__(self):
        return len(self.rgb_paths)
