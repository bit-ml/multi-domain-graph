import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
import cv2


class FilesDataset(Dataset):
    def __init__(self, dir1, dir2):
        super(FilesDataset, self).__init__()
        self.dir1 = dir1
        self.dir2 = dir2

        self.files1 = os.listdir(dir1)
        self.files2 = os.listdir(dir2)

        self.files1.sort()
        self.files2.sort()

        assert (len(self.files1) == len(self.files2))

    def __getitem__(self, index):
        f1 = np.load(os.path.join(self.dir1, self.files1[index]))
        f2 = np.load(os.path.join(self.dir2, self.files2[index]))

        return f1, f2

    def __len__(self):
        return len(self.files1)


# OUTPUT_DIR1 = "/data/multi-domain-graph-2/datasets/datasets_preproc_gt/taskonomy/tiny-train_0.15/part2/edges"
# OUTPUT_DIR2 = "/data/multi-domain-graph/datasets/datasets_preproc_ens_cu_rgb_iter1_elena/taskonomy/tiny-train_0.15/part2/edges_dexined"

# OUTPUT_DIR1 = "/data/multi-domain-graph-2/datasets/datasets_preproc_gt/taskonomy/tiny-train_0.15/part2/depth"
# OUTPUT_DIR2 = "/data/multi-domain-graph/datasets/datasets_preproc_ens_cu_rgb_iter1_elena/taskonomy/tiny-train_0.15/part2/depth_xtc"

#OUTPUT_DIR1 = "/data/multi-domain-graph-3/datasets/datasets_preproc_gt/taskonomy/tiny-test/depth"
#OUTPUT_DIR2 = "/data/multi-domain-graph-3/datasets/datasets_preproc_exp/taskonomy/tiny-test/depth_xtc"

#OUTPUT_DIR1 = "/data/multi-domain-graph-3/datasets/datasets_preproc_gt/taskonomy/tiny-val/depth"
#OUTPUT_DIR2 = "/data/multi-domain-graph-3/datasets/datasets_preproc_exp/taskonomy/tiny-val/depth_xtc"

#OUTPUT_DIR1 = "/data/multi-domain-graph-2/datasets/datasets_preproc_gt/taskonomy/tiny-train-0.15-part1/depth"
#OUTPUT_DIR2 = "/data/multi-domain-graph-2/datasets/datasets_preproc_exp/taskonomy/tiny-train-0.15-part1/depth_xtc"

#OUTPUT_DIR1 = "/data/multi-domain-graph-2/datasets/datasets_preproc_gt/taskonomy/tiny-train-0.15-part2/depth"
#OUTPUT_DIR2 = "/data/multi-domain-graph-2/datasets/datasets_preproc_exp/taskonomy/tiny-train-0.15-part2/depth_xtc"

OUTPUT_DIR1 = "/data/multi-domain-graph-2/datasets/datasets_preproc_gt/replica/val/depth"
OUTPUT_DIR2 = "/data/multi-domain-graph-6/datasets/datasets_preproc_exp/replica/val/depth_sgdepth"

OUTPUT_DIR1 = "/data/multi-domain-graph-2/datasets/datasets_preproc_gt/taskonomy/tiny-val/depth"
OUTPUT_DIR2 = "/data/multi-domain-graph-6/datasets/datasets_preproc_exp/taskonomy/tiny-val/depth_sgdepth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("OUTPUT_DIR1", OUTPUT_DIR1)
print("OUTPUT_DIR2", OUTPUT_DIR2)

dataset = FilesDataset(OUTPUT_DIR1, OUTPUT_DIR2)
data_loader = DataLoader(dataset,
                         batch_size=500,
                         shuffle=False,
                         num_workers=10)

histo_gt = np.zeros(100)
histo_exp = np.zeros(100)
with torch.no_grad():
    for batch in tqdm(data_loader):
        f1, f2 = batch

        f1 = f1.to(device=device, dtype=torch.float32)
        f2 = f2.to(device=device, dtype=torch.float32)

        f2 = 1 - f2
        f2 = torch.pow(f2, 0.5228)
        #f2 = f2 * 9

        histo_ = np.histogram(f1.cpu().numpy(), bins=100, range=(0, 1))
        histo_gt = histo_gt + histo_[0]
        histo_ = np.histogram(f2.cpu().numpy(), bins=100, range=(0, 1))
        histo_exp = histo_exp + histo_[0]

csv_file = open('res.csv', 'w')
for i in range(100):
    csv_file.write('%20.10f, %20.10f\n' % (histo_gt[i], histo_exp[i]))
csv_file.close()

histo_gt = histo_gt / (np.sum(histo_gt))
histo_exp = histo_exp / (np.sum(histo_exp))
histo_gt = np.cumsum(histo_gt)
histo_exp = np.cumsum(histo_exp)
import pdb
pdb.set_trace()
gt_med = histo_[1][np.where(histo_gt >= 0.5)[0][0]]
exp_med = histo_[1][np.where(histo_exp >= 0.5)[0][0]]

align_factor = gt_med / exp_med
gamma_factor = np.log(gt_med) / np.log(exp_med)

print("Scaling factor: %20.10f" % align_factor)
print("Gamma factor: %20.10f" % gamma_factor)
