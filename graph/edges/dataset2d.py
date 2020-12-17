import glob
import os
import pathlib
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

first_k = 3000
test_samples = 9464  #60#64
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
        cache_e1 = "%s/%s_%s_%s.npy" % (CACHE_NAME, self.experts[0].str_id,
                                        tag, pattern[-3:])
        glob_path_e1 = "%s/%s/%s/%s.npy" % (
            experts_path, self.experts[0].str_id, dataset_path, pattern)
        self.e1_output_path = load_glob_with_cache(cache_e1,
                                                   glob_path_e1)[:first_k]

        cache_e2 = "%s/%s_%s_%s.npy" % (CACHE_NAME, self.experts[1].str_id,
                                        tag, pattern[-3:])
        glob_path_e2 = "%s/%s/%s/%s.npy" % (
            experts_path, self.experts[1].str_id, dataset_path, pattern)
        self.e2_output_path = load_glob_with_cache(cache_e2,
                                                   glob_path_e2)[:first_k]
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


taskonomy_src_domains = [
    'rgb', 'depth_sgdepth', 'edges_dexined', 'normals_xtc',
    'halftone_cmyk_basic', 'halftone_gray_basic', 'halftone_rgb_basic',
    'halftone_rot_gray_basic', 'saliency_seg_egnet', 'sseg_deeplabv3',
    'sseg_fcn'
]
taskonomy_dst_domains = [
    'rgb', 'depth_sgdepth', 'edges_dexined', 'normals_xtc'
]
taskonomy_dst_domains_alt_names = [
    'rgb', 'depth_zbuffer', 'edge_texture', 'normal'
]
taskonomy_annotations_path = r'/data/multi-domain-graph/datasets/taskonomy/taskonomy-sample-model-1-master'
taskonomy_experts_path = r'/data/multi-domain-graph/datasets/taskonomy/taskonomy-sample-model-1-master-experts'


def load_filelist_with_cache(cache_file_input, cache_file_output, cache_file_pseudo_gt, src_path,
                             dst_path, pseudo_gt_dst_path, alt_name):
    if os.path.exists(cache_file_input) and os.path.exists(
            cache_file_output) and os.path.exists(cache_file_pseudo_gt):
        inputs_path = np.load(cache_file_input)
        outputs_path = np.load(cache_file_output)
        pseudo_gts_path = np.load(cache_file_pseudo_gt)
        return inputs_path, outputs_path, pseudo_gts_path

    # cache paths
    inputs_path = []
    outputs_path = []
    pseudo_gts_path = []

    filenames = os.listdir(dst_path)
    filenames.sort()
    if test_samples != 0:
        filenames = filenames[0:test_samples]
    for filename in filenames:
        # save file list for caching
        inputs_path.append(
            os.path.join(src_path, filename.replace('_%s.' % alt_name,
                                                    '_rgb.')))
        outputs_path.append(os.path.join(dst_path, filename))
        pseudo_gts_path.append(
            os.path.join(pseudo_gt_dst_path,
                         filename.replace('_%s.' % alt_name, '_rgb.')))
    # save input/output cache
    np.save(cache_file_input, np.array(inputs_path))
    np.save(cache_file_output, np.array(outputs_path))
    np.save(cache_file_pseudo_gt, np.array(pseudo_gts_path))

    return inputs_path, outputs_path, pseudo_gts_path


class DomainTestDataset(Dataset):
    """Build Testing Dataset 
    """
    def __init__(self, src_expert, dst_expert):
        super(DomainTestDataset, self).__init__()
        self.src_expert = src_expert
        self.dst_expert = dst_expert
        if src_expert in taskonomy_src_domains and dst_expert in taskonomy_dst_domains:
            self.available = True
        else:
            self.available = False
            return

        # src/dst paths
        src_path = os.path.join(taskonomy_experts_path, src_expert)
        index = taskonomy_dst_domains.index(dst_expert)
        alt_name = taskonomy_dst_domains_alt_names[index]
        dst_path_preproc = os.path.join(
            "%s-preproc" % taskonomy_annotations_path, alt_name)
        pseudo_gt_dst_path = os.path.join(taskonomy_experts_path, dst_expert)

        # save preproc files
        if not os.path.exists(dst_path_preproc):
            pathlib.Path(dst_path_preproc).mkdir(parents=True, exist_ok=True)
            dst_path = os.path.join(taskonomy_annotations_path, alt_name)
            self.__save_preprocessed__(dst_path, dst_path_preproc)

        # cache filenames
        cache_file_input = "%s/dataset_test_input_%s.npy" % (CACHE_NAME,
                                                             src_expert)
        cache_file_output = "%s/dataset_test_output_%s.npy" % (CACHE_NAME,
                                                               dst_expert)
        cache_file_pseudo_gt = "%s/dataset_test_input_%s.npy" % (CACHE_NAME,
                                                               dst_expert)

        self.inputs_path, self.outputs_path, self.pseudo_gt_outputs_path = load_filelist_with_cache(
            cache_file_input, cache_file_output, cache_file_pseudo_gt, src_path, dst_path_preproc,
            pseudo_gt_dst_path, alt_name)

    def __getitem__(self, index):
        if self.available == False:
            return None, None

        inp = np.load(self.inputs_path[index])
        outp = np.load(self.outputs_path[index])
        pseudo_gt = np.load(self.pseudo_gt_outputs_path[index])

        return inp, outp, pseudo_gt

    def __len__(self):
        if self.available == False:
            return 0
        return len(self.inputs_path)

    def __save_preprocessed__(self, dst_path_raw, dst_path_preproc):
        filenames = os.listdir(dst_path_raw)
        filenames.sort()
        if test_samples != 0:
            filenames = filenames[0:test_samples]
        for filename in filenames:
            fname_path_raw = os.path.join(dst_path_raw, filename)

            output = Image.open(fname_path_raw)
            output = np.array(output)

            if len(output.shape) == 2:
                output = output[:, :, None]
            output = torch.tensor(output, dtype=torch.float32)
            output = output.permute(2, 0, 1)
            output = torch.nn.functional.interpolate(output[None], (H, W))[0]
            if self.dst_expert == 'depth_sgdepth' or self.dst_expert == 'edges_dexined':
                output = output / 65536.0
            if self.dst_expert == 'normals_xtc' or self.dst_expert == 'rgb':
                output = output / 255.0

            # save output
            fname_path_preproc = os.path.join(dst_path_preproc,
                                              filename.replace(".png", ".npy"))

            np.save(fname_path_preproc, output)
