import os
import sys
import shutil
import h5py
import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
#import experts.edges_expert
#expert = experts.edges_expert.EdgesModel(full_expert=True)
import experts.depth_expert
import experts.normals_expert
depth_expert = experts.depth_expert.DepthModelXTC(full_expert=True)
normal_expert = experts.normals_expert.SurfaceNormalsXTC('taskonomy',
                                                         full_expert=True)
WORKING_W = 256
WORKING_H = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#rgb_files_path = r'/data/multi-domain-graph-2/datasets/Taskonomy/tiny-val/rgb'
#edge_files_path = r'/data/multi-domain-graph-2/datasets/datasets_preproc_gt/taskonomy/tiny-val/edges'

rgb_path = r'/data/multi-domain-graph-6/datasets/ml_hypersim/ai_001_003/images/scene_cam_00_final_preview'
# frame.0000.color.jpg
geo_path = r'/data/multi-domain-graph-6/datasets/ml_hypersim/ai_001_003/images/scene_cam_00_geometry_hdf5'
# frame.0000.depth_meters.hdf5
# frame.0000.normal_cam.hdf5
# frame.0000.semantic.hdf5
# frame.0000.semantic_instance.hdf5

rgb_file_path = os.path.join(rgb_path, 'frame.0000.color.jpg')

depth_file_path = os.path.join(geo_path, 'frame.0000.depth_meters.hdf5')
normal_file_path = os.path.join(geo_path, 'frame.0000.normal_cam.hdf5')
semantic_file_path = os.path.join(geo_path, 'frame.0000.semantic.hdf5')
semantic_instance_file_path = os.path.join(
    geo_path, 'frame.0000.semantic_instance.hdf5')

depth_file = h5py.File(depth_file_path, "r")
depth_info = np.array(depth_file.get('dataset')).astype('float32')
depth_info = torch.from_numpy(depth_info).unsqueeze(0)
depth_info = torch.nn.functional.interpolate(depth_info[None],
                                             (WORKING_H, WORKING_W))[0]
normal_file = h5py.File(normal_file_path, "r")
normal_info = np.array(normal_file.get('dataset')).astype('float32')
normal_info = torch.from_numpy(normal_info).permute(2, 0, 1)
normal_info = torch.nn.functional.interpolate(normal_info[None],
                                              (WORKING_H, WORKING_W))[0]
rgb = cv2.imread(rgb_file_path)
#rgb_ = rgb[0:256, 0:256, :]

rgb = cv2.resize(rgb, (WORKING_W, WORKING_H), cv2.INTER_CUBIC)
rgb_ = rgb
rgb_batch = cv2.cvtColor(rgb_, cv2.COLOR_BGR2RGB)
rgb_batch = torch.from_numpy(rgb_batch).unsqueeze(0)
import pdb
pdb.set_trace()
depth_info = depth_info / 15.625
depth_info = torch.clamp(depth_info, 0, 1)
#normal_info[2, :, :] = 1 - normal_info[2, :, :]

normal_info[1, :, :] = normal_info[1, :, :] * (-1)
normal_info[2, :, :] = normal_info[2, :, :] * (-1)
histo_x = np.histogram(normal_info[0, :, :].numpy(), bins=100,
                       range=(-1, 1))[0]
histo_y = np.histogram(normal_info[1, :, :].numpy(), bins=100,
                       range=(-1, 1))[0]
histo_z = np.histogram(normal_info[2, :, :].numpy(), bins=100,
                       range=(-1, 1))[0]

normal_info = (normal_info + 1) / 2

exp_depth = depth_expert.apply_expert_batch(rgb_batch)[0]
exp_normal = normal_expert.apply_expert_batch(rgb_batch)[0]

exp_normal_ = (exp_normal * 2) - 1
histo_exp_x = np.histogram(exp_normal_[0, :, :], bins=100, range=(-1, 1))[0]
histo_exp_y = np.histogram(exp_normal_[1, :, :], bins=100, range=(-1, 1))[0]
histo_exp_z = np.histogram(exp_normal_[2, :, :], bins=100, range=(-1, 1))[0]

depth_img = depth_info.permute(1, 2, 0)
depth_img = depth_img.repeat(1, 1, 3).numpy()

normal_img = normal_info.permute(1, 2, 0).numpy()

exp_depth_img = torch.from_numpy(exp_depth).permute(1, 2, 0).repeat(1, 1,
                                                                    3).numpy()

exp_normal_img = torch.from_numpy(exp_normal).permute(1, 2, 0).numpy()

img_0 = np.concatenate((rgb, depth_img * 255, normal_img * 255), 1)
img_1 = np.concatenate((rgb_, exp_depth_img * 255, exp_normal_img * 255), 1)
img = np.concatenate((img_0, img_1), 0)
cv2.imwrite('test.png', np.uint8(img))

csv_file = open('histo.csv', 'w')
csv_file.write('x, y, z, exp_x, exp_y, exp_z,\n')
for i in range(100):
    csv_file.write('%20.10f,' % histo_x[i])
    csv_file.write('%20.10f,' % histo_y[i])
    csv_file.write('%20.10f,' % histo_z[i])
    csv_file.write('%20.10f,' % histo_exp_x[i])
    csv_file.write('%20.10f,' % histo_exp_y[i])
    csv_file.write('%20.10f,' % histo_exp_z[i])
    csv_file.write('\n')
csv_file.close()