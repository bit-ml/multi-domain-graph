import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import torch.nn.functional as F


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


def build_mask(target, val=0.0, tol=1e-3, kernel=1):
    padding = (kernel - 1) // 2
    if target.shape[1] == 1:
        mask = ((target >= val - tol) & (target <= val + tol))
        mask = F.conv2d(mask.float(),
                        torch.ones(1, 1, kernel, kernel, device=mask.device),
                        padding=padding) != 0
        return (~mask).expand_as(target)

    mask1 = (target[:, 0, :, :] >= val - tol) & (target[:, 0, :, :] <=
                                                 val + tol)
    mask2 = (target[:, 1, :, :] >= val - tol) & (target[:, 1, :, :] <=
                                                 val + tol)
    mask3 = (target[:, 2, :, :] >= val - tol) & (target[:, 2, :, :] <=
                                                 val + tol)
    mask = (mask1 & mask2 & mask3).unsqueeze(1)
    mask = F.conv2d(mask.float(),
                    torch.ones(1, 1, kernel, kernel, device=mask.device),
                    padding=padding) != 0
    return (~mask).expand_as(target)


OUTPUT_DIR1 = "/data/multi-domain-graph-5/datasets/datasets_preproc_gt/hypersim/train1/normals"
OUTPUT_DIR2 = "/data/multi-domain-graph-5/datasets/datasets_preproc_exp/hypersim/train1/normals_xtc"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("OUTPUT_DIR1", OUTPUT_DIR1)
print("OUTPUT_DIR2", OUTPUT_DIR2)

dataset = FilesDataset(OUTPUT_DIR1, OUTPUT_DIR2)
data_loader = DataLoader(dataset,
                         batch_size=500,
                         shuffle=False,
                         num_workers=10)
loss_l2 = nn.MSELoss()
loss_l1 = nn.L1Loss()

all_l1_loss = 0
all_l2_loss = 0

with torch.no_grad():
    for batch in tqdm(data_loader):
        f1, f2 = batch

        f1 = f1.to(device=device, dtype=torch.float32)
        f2 = f2.to(device=device, dtype=torch.float32)

        mask = build_mask(f1, 2, tol=1e-3)
        #f1 = f1 * mask
        #f2 - f2 * mask

        l2_loss = loss_l2(f1, f2)
        l1_loss = loss_l1(f1, f2)

        all_l2_loss += l2_loss.item()
        all_l1_loss += l1_loss.item()

print("L1 loss %.2f" % (all_l1_loss / len(data_loader) * 100))
print("L2 loss %.2f" % (all_l2_loss / len(data_loader) * 100))
