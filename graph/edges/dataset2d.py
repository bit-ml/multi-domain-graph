import glob
import os
import pathlib
import time
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

first_k = 3000
test_samples = 9464#60#64

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


taskonomy_src_domains = [
    'rgb', 'depth_sgdepth', 'edges_dexined', 'normals_xtc',
    'halftone_cmyk_basic', 'halftone_gray_basic', 'halftone_rgb_basic', 'halftone_rot_gray_basic',
    'saliency_seg_egnet', 'sseg_deeplabv3', 'sseg_fcn']
taskonomy_dst_domains = [
    'rgb', 'depth_sgdepth', 'edges_dexined', 'normals_xtc']
taskonomy_dst_domains_alt_names = [
    'rgb', 'depth_zbuffer', 'edge_texture', 'normal']
taskonomy_annotations_path = r'/data/multi-domain-graph/datasets/taskonomy/taskonomy-sample-model-1-master'
taskonomy_experts_path = r'/data/multi-domain-graph/datasets/taskonomy/taskonomy-sample-model-1-master-experts'

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
        src_path = os.path.join(taskonomy_experts_path, src_expert)
        index = taskonomy_dst_domains.index(dst_expert)
        alt_name = taskonomy_dst_domains_alt_names[index]
        dst_path = os.path.join(taskonomy_annotations_path, alt_name)
        self.inputs_path = []
        self.outputs_path = []
        filenames = os.listdir(dst_path)
        filenames.sort()
        if test_samples!=0:
            filenames = filenames[0:test_samples]
        for filename in filenames:
            self.outputs_path.append(os.path.join(dst_path, filename))
            self.inputs_path.append(os.path.join(src_path, filename.replace('_%s.png'%alt_name, '_rgb.npy')))
        
            
    def __getitem__(self, index):
        if self.available == False:
            return None, None
        input_path = self.inputs_path[index]
        output_path = self.outputs_path[index]

        input = np.load(input_path)

        output = Image.open(output_path)
        output = np.array(output)
        if len(output.shape) == 2:
            output = output[:, :, None]
        output = torch.tensor(output, dtype=torch.float32)
        output = output.permute(2, 0, 1)
        output = torch.nn.functional.interpolate(output[None], (input.shape[1], input.shape[2]))[0]
        if self.dst_expert == 'depth_sgdepth' or self.dst_expert == 'edges_dexined':
            output = output / 65536.0
        if self.dst_expert == 'normals_xtc' or self.dst_expert == 'rgb':
            output = output / 255.0
        
        return input, output

    def __len__(self):
        if self.available == False:
            return 0
        else:
            return len(self.inputs_path)