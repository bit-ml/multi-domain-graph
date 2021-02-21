import os
import shutil
import sys
import torch
import numpy as np
import torchvision
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

logs_out_path = r'/data/multi-domain-graph-6/datasets/hypersim/runs'

experts_path = r'/data/multi-domain-graph-5/datasets/datasets_preproc_exp/hypersim'
gt_path = r'/data/multi-domain-graph-5/datasets/datasets_preproc_gt/hypersim'
split_name = 'test'
n_samples = 100

experts_path = os.path.join(experts_path, split_name)
gt_path = os.path.join(gt_path, split_name)

ref_path = None
all_experts = []
if os.path.exists(experts_path):
    all_experts = os.listdir(experts_path)
    ref_path = os.path.join(experts_path, all_experts[0])
all_gts = []
if os.path.exists(gt_path):
    all_gts = os.listdir(gt_path)
    ref_path = os.path.join(gt_path, all_gts[0])

if not ref_path == None:
    imgs = os.listdir(ref_path)
    n_imgs = len(imgs)
    indexes = np.arange(0, n_imgs)
    np.random.shuffle(indexes)
    indexes = indexes[0:min(n_samples, n_imgs)]
else:
    indexes = np.arange(0, n_samples)

os.makedirs(logs_out_path, exist_ok=True)
writer = SummaryWriter(
    os.path.join(logs_out_path, split_name + '_' + str(datetime.now())))

for idx in indexes:
    for exp in all_experts:
        path = os.path.join(experts_path, exp, '%08d.npy' % idx)
        v = torch.from_numpy(np.load(path))
        img_grid = torchvision.utils.make_grid(v[None], 1)
        writer.add_image('experts/%s' % (exp), img_grid, idx)
    for gt in all_gts:
        path = os.path.join(gt_path, gt, '%08d.npy' % idx)
        v = torch.from_numpy(np.load(path))
        img_grid = torchvision.utils.make_grid(v[None], 1)
        writer.add_image('gts/%s' % (gt), img_grid, idx)
