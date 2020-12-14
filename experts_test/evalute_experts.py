import os
import shutil
import sys
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
#import torch.utils.data as data

working_h = 256
working_w = 256

experts_names = ['depth_sgdepth', 'edges_dexined', 'normals_xtc']

annotations_names = ['depth_zbuffer', 'edge_texture', 'normal']

rgbs_path = r'/data/multi-domain-graph/datasets/taskonomy/taskonomy-sample-model-1-master/rgb'
annotations_path = r'/data/multi-domain-graph/datasets/taskonomy/taskonomy-sample-model-1-master'
results_path = r'/data/multi-domain-graph/datasets/taskonomy/taskonomy-sample-model-1-master-experts'


def get_image(img_path, height, width):
    img = cv2.imread(img_path)

    img = cv2.resize(img, (width, height), cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = torch.tensor(img, dtype=torch.float32)
    img = img.permute(2, 0, 1)
    return img


def get_label(label_path, height, width, task_name):

    data = Image.open(label_path)

    data = np.array(data)

    if len(data.shape) == 2:
        data = data[:, :, None]
    data = torch.tensor(data, dtype=torch.float32)
    data = data.permute(2, 0, 1)
    data = torch.nn.functional.interpolate(data[None], (height, width))[0]

    if task_name == 'depth_sgdepth' or task_name == 'edges_dexined':
        data = data / 65536.0
    if task_name == 'normals_xtc':
        data = data / 255.0

    return data


def get_results(results_path, filenames):
    all_results = []
    for filename in filenames:
        res = np.load(
            os.path.join(results_path, filename.replace('.png', '.npy')))
        res = torch.tensor(res, dtype=torch.float32)
        if len(res.shape) == 2:
            res = res[None]
        all_results.append(res[None])
    return torch.cat(all_results, 0)


class TaskTestDataset_ImgLevel_Taskonomy(Dataset):
    """
    Testing Dataset for image-level tasks, from Taskonomy dataset 
    """
    def __init__(self, rgbs_path, annotations_path, task_name, task_name_in_db,
                 height, width):
        """
            Parameters:
            -----------
            rgbs_path - path to rgb data (folder containing all testing images)
            annotations_path - path to a folder containing task annotations 
            task_name - considered task
            task_name_in_db - name used inside the db for current task 
        """
        super(TaskTestDataset_ImgLevel_Taskonomy, self).__init__()
        self.task_name = task_name

        self.height = height
        self.width = width

        filenames = os.listdir(rgbs_path)
        filenames.sort()
        self.imgs_path = []
        self.annotations_path = []
        self.filenames = filenames
        for filename in filenames:
            self.imgs_path.append(os.path.join(rgbs_path, filename))
            self.annotations_path.append(
                os.path.join(annotations_path,
                             filename.replace('rgb', task_name_in_db)))

    def __getitem__(self, index):
        img = get_image(self.imgs_path[index], self.height, self.width)
        label = get_label(self.annotations_path[index], self.height,
                          self.width, self.task_name)
        return img, label, self.filenames[index]

    def __len__(self):
        return len(self.imgs_path)


def evaluate_task(task_dataset, task_name, results_path):
    test_dataloader = torch.utils.data.DataLoader(task_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  drop_last=False)
    task_results_path = os.path.join(results_path, task_name)
    task_l1_loss = 0
    task_l2_loss = 0
    for batch_idx, (imgs, labels, filenames) in enumerate(test_dataloader):
        print(batch_idx)
        results = get_results(task_results_path, filenames)

        l1_loss = torch.nn.functional.l1_loss(results,
                                              labels,
                                              reduction='none').view(
                                                  batch_size, -1)
        l2_loss = torch.nn.functional.mse_loss(results,
                                               labels,
                                               reduction='none').view(
                                                   batch_size, -1)

        l1_loss = torch.mean(l1_loss, 1)
        l2_loss = torch.mean(l2_loss, 1)
        task_l1_loss = task_l1_loss + l1_loss.sum()
        task_l2_loss = task_l2_loss + l2_loss.sum()
    n_samples = task_dataset.__len__()
    task_l1_loss = task_l1_loss / n_samples
    task_l2_loss = task_l2_loss / n_samples
    print('%s: l1 %20.10f -- l2 %20.10f' %
          (task_name, task_l1_loss, task_l2_loss))


if __name__ == "__main__":
    batch_size = 8

    depth_test_dataset = TaskTestDataset_ImgLevel_Taskonomy(
        rgbs_path, os.path.join(annotations_path, 'depth_zbuffer'),
        'depth_sgdepth', 'depth_zbuffer', working_h, working_w)
    evaluate_task(depth_test_dataset, 'depth_sgdepth', results_path)

    edges_test_dataset = TaskTestDataset_ImgLevel_Taskonomy(
        rgbs_path, os.path.join(annotations_path, 'edge_texture'),
        'edges_dexined', 'edge_texture', working_h, working_w)
    evaluate_task(edges_test_dataset, 'edges_dexined', results_path)

    normals_test_dataset = TaskTestDataset_ImgLevel_Taskonomy(
        rgbs_path, os.path.join(annotations_path, 'normal'), 'normals_xtc',
        'normal', working_h, working_w)
    evaluate_task(normals_test_dataset, 'normals_xtc', results_path)
