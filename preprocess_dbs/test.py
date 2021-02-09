import sys
import shutil
import os
import numpy as np
import cv2
import torch
import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
path_depth = r'/data/multi-domain-graph-3/datasets/datasets_preproc_gt/taskonomy/tiny-test-ok/depth'
path_depth_exp = r'/data/multi-domain-graph-3/datasets/datasets_preproc_exp/taskonomy/tiny-test-ok/depth_xtc'
path_edges = r'/data/multi-domain-graph-3/datasets/datasets_preproc_gt/taskonomy/tiny-test-ok/edges'
path_edges_exp = r'/data/multi-domain-graph-3/datasets/datasets_preproc_exp/taskonomy/tiny-test-ok/edges_dexined'

path_normals = r'/data/multi-domain-graph-3/datasets/datasets_preproc_gt/taskonomy/tiny-test-ok/normals'
path_rgb = r'/data/multi-domain-graph-3/datasets/datasets_preproc_exp/taskonomy/tiny-test-ok/rgb'
path_halftone = r'/data/multi-domain-graph-3/datasets/datasets_preproc_exp/taskonomy/tiny-test-ok/halftone_gray_basic'
path = r'/data/multi-domain-graph-3/datasets/datasets_preproc_exp/taskonomy/tiny-test-ok/saliency_seg_egnet'


class Domain2DDataset(Dataset):
    def __init__(self, path):
        super(Domain2DDataset, self).__init__()
        self.paths = os.listdir(path)
        self.paths.sort()
        self.main_path = path

    def __getitem__(self, index):
        v = np.load(os.path.join(self.main_path, self.paths[index]))
        return v

    def __len__(self):
        return len(self.paths)


#dataset = Domain2DDataset(path_edges_exp)
#dataset = Domain2DDataset(path_depth_exp)
#dataset = Domain2DDataset(path_depth)
dataset = Domain2DDataset(path_edges_exp)

loader = DataLoader(dataset, batch_size=500, shuffle=False, num_workers=10)
histo = np.zeros(20)
batches = 0
num_batches = len(loader)
for batch in loader:
    n_elems = np.prod(batch.shape)
    batch = batch.view(n_elems).numpy()
    histo_ = np.histogram(batch, bins=20, range=(0.01, 1))
    histo = histo + histo_[0]
    print(histo_[1])

#histo = histo / dataset.__len__()

print(histo)
histo = histo / sum(histo)

for i in range(20):
    print("%5.4f " % histo[i])
'''
files = os.listdir(path_edges)
files.sort()
histo = np.zeros(20)
for file_ in files:
    v = np.load(os.path.join(path_edges, file_))
    histo_ = np.histogram(v, bins=20, range=(0, 1))
    histo = histo + histo_[0]
histo = histo / len(files)
'''
'''
files = os.listdir(path)
files.sort()
idx = 0
files = [files[39], files[50], files[53]]
for file_ in files:
    print(file_)
    v_seg = np.load(os.path.join(path, file_))
    v_seg = v_seg / np.max(v_seg)
    v_seg = np.moveaxis(v_seg, 0, -1)
    cv2.imwrite('saliency_seg_%d.png' % idx, np.uint8(v_seg * 255))

    v_rgb = np.load(os.path.join(path_rgb, file_))
    v_rgb = np.moveaxis(v_rgb, 0, -1)
    cv2.imwrite('rgb_%d.png' % idx, np.uint8(v_rgb * 255))

    v = np.load(os.path.join(path_halftone, file_))
    v = np.moveaxis(v, 0, -1)
    cv2.imwrite('halftone_%d.png' % idx, np.uint8(v * 255))

    v = np.load(os.path.join(path_edges, file_))
    v = np.moveaxis(v, 0, -1)
    cv2.imwrite('edges_%d.png' % idx, np.uint8(v * 255))

    v = np.load(os.path.join(path_depth, file_))
    v = np.moveaxis(v, 0, -1)
    cv2.imwrite('depth_%d.png' % idx, np.uint8(v * 255))

    v = np.load(os.path.join(path_normals, file_))
    v = np.moveaxis(v, 0, -1)
    cv2.imwrite('normals_%d.png' % idx, np.uint8(v * 255))

    idx = idx + 1
'''
'''
files = os.listdir(path)
files.sort()
min_v = 20
max_v = 0
for file_ in files:
    v_seg = np.load(os.path.join(path, file_))
    min_v_ = np.min(v_seg)
    max_v_ = np.max(v_seg)
    max_v = max(max_v, max_v_)
    min_v = min(min_v, min_v_)

    print('min: %20.10f -- max: %20.10f' % (min_v, max_v))
'''
