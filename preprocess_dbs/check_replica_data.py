import os
import shutil
import sys
import torch
import numpy as np
import torchvision
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

#logs_out_path = r'/data/multi-domain-graph-6/datasets/replica_raw_1/runs'
#logs_out_path = r'/data/multi-domain-graph-2/datasets/replica_raw_2/runs'
logs_out_path = r'/data/multi-domain-graph-2/datasets/replica_raw_2/runs'

#experts_path = r'/data/multi-domain-graph-2/datasets/datasets_preproc_exp/replica'
#gt_path = r'/data/multi-domain-graph-2/datasets/datasets_preproc_gt/replica'
experts_path = r'/data/multi-domain-graph-4/datasets/datasets_preproc_ens_iter1/replica_2'
gt_path = r'/data/multi-domain-graph-4/datasets/datasets_preproc_ens_iter1/replica'
split_name = 'train'
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

import pdb
pdb.set_trace()
for idx in indexes:
    for exp in all_experts:
        if exp == 'rgb' or exp == 'hsv' or exp == 'halftone_gray' or exp == 'grayscale':
            experts_path_ = r'/data/multi-domain-graph-2/datasets/datasets_preproc_exp/replica_2/train'
        else:
            experts_path_ = experts_path
            #continue
        path = os.path.join(experts_path_, exp, '%08d.npy' % idx)
        if not os.path.exists(path):
            path = os.path.join(experts_path_, exp, '%05d.npy' % idx)
        v = torch.from_numpy(np.load(path))
        img_grid = torchvision.utils.make_grid(v[None], 1)
        writer.add_image('experts/%s' % (exp), img_grid, idx)
    for gt in all_gts:
        if gt == 'rgb' or gt == 'hsv' or gt == 'halftone_gray' or gt == 'grayscale':
            gt_path_ = r'/data/multi-domain-graph-2/datasets/datasets_preproc_exp/replica/train'
        else:
            gt_path_ = gt_path
            #continue
        path = os.path.join(gt_path_, gt, '%08d.npy' % idx)
        if not os.path.exists(path):
            path = os.path.join(gt_path_, gt, '%05d.npy' % idx)
        v = torch.from_numpy(np.load(path))
        v[v > 1] = 1
        img_grid = torchvision.utils.make_grid(v[None], 1)
        writer.add_image('gts/%s' % (gt), img_grid, idx)
