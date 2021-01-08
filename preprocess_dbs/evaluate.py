import os
import sys
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

EXPERTS_PATH = r'/data/multi-domain-graph/datasets/datasets_preproc_exp'
GT_PATH = r'/data/multi-domain-graph/datasets/datasets_preproc_gt'

DATASET_PATH = r'taskonomy/sample-model'

usage_str = 'python evaluate.py dataset_path'


class TestDataset_ImgLevel(Dataset):
    def __init__(self, exp_path, gt_path):
        super(TestDataset_ImgLevel, self).__init__()
        filenames = os.listdir(exp_path)
        filenames.sort()
        self.exp_paths = []
        self.gt_paths = []

        for filename in filenames:
            self.exp_paths.append(os.path.join(exp_path, filename))
            self.gt_paths.append(os.path.join(gt_path, filename))

    def __getitem__(self, index):
        exp_res = np.load(self.exp_paths[index])
        gt_res = np.load(self.gt_paths[index])
        exp_res = torch.tensor(exp_res, dtype=torch.float32)
        gt_res = torch.tensor(gt_res, dtype=torch.float32)
        return exp_res, gt_res

    def __len__(self):
        return len(self.exp_paths)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit('incorrect usage [%s]' % usage_str)
    if not sys.argv[1] == '-':
        DATASET_PATH = sys.argv[1]

    db_exp_path = os.path.join(EXPERTS_PATH, DATASET_PATH)
    db_gt_path = os.path.join(GT_PATH, DATASET_PATH)

    gts = os.listdir(db_gt_path)
    experts = os.listdir(db_exp_path)

    #experts = ['normals_xtc']

    for exp_name in experts:
        gt_name = None
        for i in range(len(gts)):
            gt_name_ = gts[i]
            if exp_name[0:len(gt_name_)] == gt_name_:
                gt_name = gts[i]
        if gt_name == None:
            continue
        print('%s vs. GT %s' % (exp_name, gt_name))
        exp_results_path = os.path.join(db_exp_path, exp_name)
        gt_results_path = os.path.join(db_gt_path, gt_name)
        dataset = TestDataset_ImgLevel(exp_results_path, gt_results_path)
        batch_size = 30
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 drop_last=False,
                                                 num_workers=8)
        task_l1_loss = 0
        task_l2_loss = 0
        n_batches = np.round(dataset.__len__() / batch_size) + 1

        for batch_idx, (exp_res, gt_res) in enumerate(tqdm(dataloader)):

            l1_loss = torch.nn.functional.l1_loss(exp_res,
                                                  gt_res,
                                                  reduction='none').view(
                                                      exp_res.shape[0], -1)
            l2_loss = torch.nn.functional.mse_loss(exp_res,
                                                   gt_res,
                                                   reduction='none').view(
                                                       exp_res.shape[0], -1)

            l1_loss = torch.mean(l1_loss, 1)
            l2_loss = torch.mean(l2_loss, 1)
            task_l1_loss = task_l1_loss + l1_loss.sum()
            task_l2_loss = task_l2_loss + l2_loss.sum()

        n_samples = dataset.__len__()
        task_l1_loss = task_l1_loss / n_samples
        task_l2_loss = task_l2_loss / n_samples
        print('%20s vs %20s: L1 %20.10f L2 %20.10f' %
              (exp_name, gt_name, task_l1_loss, task_l2_loss))