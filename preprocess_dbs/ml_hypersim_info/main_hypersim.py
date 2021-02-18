import os
import sys
import shutil
import cv2
import torch
import h5py
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# semantic labels - NYU 40
# from https://github.com/ankurhanda/SceneNetv1.0
semantic_labels = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
    'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves',
    'curtain', 'dresser', 'pillow', 'mirror', 'floor-mat', 'clothes',
    'ceiling', 'books', 'refrigerator', 'television', 'paper', 'towel',
    'shower-curtain', 'box', 'whiteboard', 'person', 'nightstand', 'toilet',
    'sink', 'lamp', 'bathtub', 'bag', 'other-structure', 'other-furniture',
    'other-prop'
]

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import experts.halftone_expert
import experts.grayscale_expert
import experts.hsv_expert
import experts.normals_expert
import experts.depth_expert
import experts.edges_expert

usage_str = 'usage: python main_hypersim.py type split_name exp1 exp2 ...'
#    type                   - [0/1] - 0 create preprocessed gt samples
#                                   - 1 create preprocessed experts samples
#    split_name              - train/valid/test
#    expi                   - name of the i'th expert / domain
#                           - should be one of the VALID_EXPERTS_NAME / VALID_GT_DOMAINS
#                           - 'all' to run all available experts / domains

VALID_EXPERTS_NAME = ['depth_xtc', 'normals_xtc', 'edges_dexined']
VALID_DOMAINS_NAME = [
    'rgb', 'grayscale', 'hsv', 'halftone_gray', 'depth', 'normals', 'sem_seg'
]

COMPUTED_GT_DOMAINS = ['grayscale', 'hsv', 'halftone_gray']

VALID_SPLITS_NAME = ['train1', 'train2', 'train3', 'test', 'valid']

WORKING_H = 256
WORKING_W = 256

EXP_OUT_PATH_ = r'/data/multi-domain-graph-5/datasets/datasets_preproc_exp/hypersim'
GT_OUT_PATH_ = r'/data/multi-domain-graph-5/datasets/datasets_preproc_gt/hypersim'
DATASET_PATH = r'/data/multi-domain-graph-6/datasets/ml_hypersim/data'
SPLITS_CSV_PATH = r'/data/multi-domain-graph-6/datasets/ml_hypersim/metadata_images_split_scene_v1_selection.csv'

RUN_TYPE = 0
EXPERTS_NAME = []
DOMAINS_NAME = []
EXP_OUT_PATH = ''
GT_OUT_PATH = ''
SPLIT_NAME = ''


def check_arguments_without_delete(argv):
    global RUN_TYPE
    global EXPERTS_NAME
    global DOMAINS_NAME
    global EXP_OUT_PATH
    global GT_OUT_PATH
    global SPLIT_NAME
    if len(argv) < 4:
        return 0, 'incorrect usage'

    RUN_TYPE = np.int32(argv[1])
    if not (RUN_TYPE == 0 or RUN_TYPE == 1):
        return 0, 'incorrect run type: %d' % RUN_TYPE

    SPLIT_NAME = argv[2]
    if SPLIT_NAME not in VALID_SPLITS_NAME:
        status = 0
        status_code = 'Split %s is not valid' % SPLIT_NAME
        return status, status_code
    print('SPLIT:', SPLIT_NAME)

    EXP_OUT_PATH = os.path.join(EXP_OUT_PATH_, SPLIT_NAME)
    GT_OUT_PATH = os.path.join(GT_OUT_PATH_, SPLIT_NAME)

    if RUN_TYPE == 0:
        if argv[3] == 'all':
            DOMAINS_NAME = VALID_DOMAINS_NAME
        else:
            potential_domains = argv[3:]
            print("potential_domains", potential_domains)
            print("valid domains", VALID_DOMAINS_NAME)
            DOMAINS_NAME = []

            for i in range(len(potential_domains)):
                dom_name = potential_domains[i]
                if not dom_name in VALID_DOMAINS_NAME:
                    status = 0
                    status_code = 'Domain %s is not valid' % dom_name
                    return status, status_code
                DOMAINS_NAME.append(dom_name)
        print("DOMAINS:", DOMAINS_NAME)
        return 1, ''
    else:
        if argv[3] == 'all':
            EXPERTS_NAME = VALID_EXPERTS_NAME
        else:
            potential_experts = argv[3:]
            EXPERTS_NAME = []
            for i in range(len(potential_experts)):
                exp_name = potential_experts[i]
                if not exp_name in VALID_EXPERTS_NAME:
                    status = 0
                    status_code = 'Expert %s is not valid' % exp_name
                    return status, status_code
                EXPERTS_NAME.append(exp_name)
        return 1, ''


def get_task_split_paths(dataset_path, splits_csv_path, split_name, folder_str,
                         task_str, ext):
    if split_name == "valid":
        split_name = "val"
    train_index = -1
    if split_name.find('train') != -1:
        train_index = int(split_name[-1])
        split_name = 'train'
    df = pd.read_csv(splits_csv_path)
    df = df[df['included_in_public_release'] == True]
    df = df[df['split_partition_name'] == split_name]

    paths = []
    scenes = df['scene_name'].unique()
    scenes = scenes[0:1]
    for scene in scenes:
        df_scene = df[df['scene_name'] == scene]
        cameras = df_scene['camera_name'].unique()
        for camera in cameras:
            df_camera = df_scene[df_scene['camera_name'] == camera]
            frames = df_camera['frame_id'].unique()
            for frame in frames:
                path = '%s/%s/images/scene_%s_%s/frame.%04d.%s.%s' % (
                    dataset_path, scene, camera, folder_str, int(frame),
                    task_str, ext)
                paths.append(path)

    if train_index >= 0:
        n_samples = len(paths)
        n_set_samples = n_samples // 3
        first = n_set_samples + (n_samples - 3 * n_set_samples)
        second = first + n_set_samples
        third = second + n_set_samples
        if train_index == 1:
            paths = paths[0:first]
        elif train_index == 2:
            paths = paths[first:second]
        elif train_index == 3:
            paths = paths[second:third]
        else:
            paths = []
    return paths


class RGBDataset(Dataset):
    def __init__(self, dataset_path, splits_csv_path, split_name):
        super(RGBDataset, self).__init__()
        self.paths = get_task_split_paths(dataset_path, splits_csv_path,
                                          split_name, 'final_preview',
                                          'tonemap', 'jpg')

    def __getitem__(self, index):
        rgb_info = cv2.imread(self.paths[index])
        rgb_info = cv2.resize(rgb_info, (WORKING_W, WORKING_H),
                              cv2.INTER_CUBIC)
        rgb_info = cv2.cvtColor(rgb_info, cv2.COLOR_BGR2RGB)
        rgb_info = np.moveaxis(rgb_info, -1, 0)
        rgb_info = rgb_info.astype('float32') / 255
        return rgb_info, self.paths[index]

    def __len__(self):
        return len(self.paths)


class RGBDataset_ForExperts(Dataset):
    def __init__(self, dataset_path, splits_csv_path, split_name):
        super(RGBDataset_ForExperts, self).__init__()
        self.paths = get_task_split_paths(dataset_path, splits_csv_path,
                                          split_name, 'final_preview',
                                          'tonemap', 'jpg')

    def __getitem__(self, index):
        rgb_info = cv2.imread(self.paths[index])
        rgb_info = cv2.resize(rgb_info, (WORKING_W, WORKING_H),
                              cv2.INTER_CUBIC)
        rgb_info = cv2.cvtColor(rgb_info, cv2.COLOR_BGR2RGB)
        rgb_info = rgb_info.astype('float32')
        return rgb_info, self.paths[index]

    def __len__(self):
        return len(self.paths)


class DepthDataset(Dataset):
    def __init__(self, dataset_path, splits_csv_path, split_name):
        super(DepthDataset, self).__init__()
        self.paths = get_task_split_paths(dataset_path, splits_csv_path,
                                          split_name, 'geometry_hdf5',
                                          'depth_meters', 'hdf5')

    def __getitem__(self, index):
        depth_file = h5py.File(self.paths[index], "r")
        depth_info = np.array(depth_file.get('dataset')).astype('float32')
        depth_info = torch.from_numpy(depth_info).unsqueeze(0)
        depth_info = torch.nn.functional.interpolate(depth_info[None],
                                                     (WORKING_H, WORKING_W))[0]
        depth_info[depth_info != depth_info] = 0  # get rid of nan values
        depth_info = depth_info / 31.25
        depth_info = torch.clamp(depth_info, 0, 1)
        return depth_info, self.paths[index]

    def __len__(self):
        return len(self.paths)


class NormalsDataset(Dataset):
    def __init__(self, dataset_path, splits_csv_path, split_name):
        super(NormalsDataset, self).__init__()
        self.paths = get_task_split_paths(dataset_path, splits_csv_path,
                                          split_name, 'geometry_hdf5',
                                          'normal_cam', 'hdf5')

    def __getitem__(self, index):
        normal_file = h5py.File(self.paths[index], "r")
        normal_info = np.array(normal_file.get('dataset')).astype('float32')
        normal_info = torch.from_numpy(normal_info).permute(2, 0, 1)
        normal_info = torch.nn.functional.interpolate(
            normal_info[None], (WORKING_H, WORKING_W))[0]

        normal_info[1, :, :] = normal_info[1, :, :] * (-1)
        normal_info[2, :, :] = normal_info[2, :, :] * (-1)

        norm = torch.norm(normal_info, dim=0, keepdim=True)
        normal_info = normal_info / norm

        normal_info = (normal_info + 1) / 2

        return normal_info, self.paths[index]

    def __len__(self):
        return len(self.paths)


class SemanticSegDataset(Dataset):
    def __init__(self, dataset_path, splits_csv_path, split_name):
        super(SemanticSegDataset, self).__init__()
        self.paths = get_task_split_paths(dataset_path, splits_csv_path,
                                          split_name, 'geometry_hdf5',
                                          'semantic', 'hdf5')

    def __getitem__(self, index):
        semantic_file = h5py.File(self.paths[index], "r")
        semantic_info = np.array(semantic_file.get('dataset'))
        semantic_info[semantic_info == -1] = 0
        semantic_info = np.uint8(semantic_info)

        semantic_info = cv2.resize(semantic_info, (WORKING_W, WORKING_H),
                                   cv2.INTER_NEAREST)
        semantic_info = semantic_info.astype('float32')
        semantic_info[semantic_info == -1] = 0
        semantic_info = semantic_info[None]
        return semantic_info, self.paths[index]

    def __len__(self):
        return len(self.paths)


def get_expert(exp_name):
    if exp_name == 'depth_xtc':
        return experts.depth_expert.DepthModelXTC(full_expert=True)
    elif exp_name == 'normals_xtc':
        return experts.normal_expert.SurfaceNormalsXTC(dataset_name='hypersim',
                                                       full_expert=True)
    elif exp_name == 'edges_dexined':
        return experts.edges_expert.EdgesModel(full_expert=True)

    return None


def add_depth_process(data):
    return data / 2


def get_exp_results():
    print("get experts ", EXPERTS_NAME)

    dataset = RGBDataset_ForExperts(DATASET_PATH, SPLITS_CSV_PATH, SPLIT_NAME)
    dataloader = DataLoader(dataset,
                            batch_size=30,
                            shuffle=False,
                            num_workers=20)
    for exp in EXPERTS_NAME:
        exp_out_path = os.path.join(EXP_OUT_PATH, exp)
        os.makedirs(exp_out_path, exist_ok=True)
        expert = get_expert(exp)
        process_fct = expert.apply_expert_batch
        if exp == 'depth_xtc':
            add_process_fct = add_depth_process
        else:
            add_process_fct = lambda x: x
        file_idx = 0
        for batch in tqdm(dataloader):
            exp_info, paths = batch
            exp_info = process_fct(exp_info)
            exp_info = add_process_fct(exp_info)
            for i in range(exp_info.shape[0]):
                exp_info_ = exp_info[i]
                exp_info_ = np.array(exp_info_)
                np.save(os.path.join(exp_out_path, '%08d.npy' % file_idx),
                        exp_info_)
                file_idx += 1
    return


def get_expert_gt(dom_name):
    if dom_name == 'grayscale':
        return experts.grayscale_expert.Grayscale(full_expert=True)
    elif dom_name == 'hsv':
        return experts.hsv_expert.HSVExpert(full_expert=True)
    elif dom_name == 'halftone_gray':
        return experts.halftone_expert.HalftoneModel(full_expert=True, style=0)
    return None


def get_dataset_type(dom_name):
    if dom_name in COMPUTED_GT_DOMAINS:
        return RGBDataset_ForExperts
    elif dom_name == 'rgb':
        return RGBDataset
    elif dom_name == 'depth':
        return DepthDataset
    elif dom_name == 'normals':
        return NormalsDataset
    elif dom_name == 'sem_seg':
        return SemanticSegDataset
    return None


def get_gt_domains():
    print("get_gt_domains", DOMAINS_NAME)

    for dom in DOMAINS_NAME:
        dom_out_path = os.path.join(GT_OUT_PATH, dom)
        os.makedirs(dom_out_path, exist_ok=True)

        if dom in COMPUTED_GT_DOMAINS:
            expert = get_expert_gt(dom)
            process_fct = expert.apply_expert_batch
        else:
            process_fct = lambda x: x

        task_dataset_cls = get_dataset_type(dom)
        dataset = task_dataset_cls(DATASET_PATH, SPLITS_CSV_PATH, SPLIT_NAME)
        dataloader = DataLoader(dataset,
                                batch_size=100,
                                shuffle=False,
                                num_workers=20)

        file_idx = 0
        for batch in tqdm(dataloader):
            dom_info, paths = batch
            dom_info = process_fct(dom_info)

            for i in range(dom_info.shape[0]):
                dom_info_ = dom_info[i]
                dom_info_ = np.array(dom_info_)
                np.save(os.path.join(dom_out_path, '%08d.npy' % file_idx),
                        dom_info_)
                file_idx += 1


def check_wrong_depth_maps(dataloader):
    min_values = []
    max_values = []
    for batch in tqdm(dataloader):
        depth_info, paths = batch
        nan_mask = depth_info != depth_info
        non_nan_mask = ~nan_mask
        min_values.append(torch.min(depth_info[non_nan_mask]))
        max_values.append(torch.max(depth_info[non_nan_mask]))
        '''
        if torch.max(depth_info[non_nan_mask] > 50):

            for i in range(depth_info.shape[0]):

                d_i = depth_info[i]
                nan_mask_ = d_i != d_i
                non_nan_mask_ = ~nan_mask_
                if torch.max(d_i[non_nan_mask_] > 500):
                    print(paths[i])
        '''
    min_val = np.min(np.array(min_values))
    max_val = np.max(np.array(max_values))
    print('Depth: min %8.4f -- max %8.4f' % (min_val, max_val))


def check_normals_range(split_name):
    dataset = NormalsDataset(DATASET_PATH, SPLITS_CSV_PATH, split_name)
    dataloader = DataLoader(dataset,
                            batch_size=100,
                            shuffle=False,
                            num_workers=20)
    print('split %s - %d' % (split_name, dataset.__len__()))
    min_values = []
    max_values = []
    histo_x = np.zeros(200)
    histo_y = np.zeros(200)
    histo_z = np.zeros(200)
    #import pdb
    #pdb.set_trace()
    for batch in tqdm(dataloader):
        data, paths = batch
        #import pdb
        #pdb.set_trace()
        nan_mask = data != data
        non_nan_mask = ~nan_mask
        min_values.append(torch.min(data[non_nan_mask]))
        max_values.append(torch.max(data[non_nan_mask]))

        data_x = data[:, 0, :, :]
        nan_mask = data_x != data_x
        non_nan_mask = ~nan_mask
        histo_x_ = np.histogram(data_x[non_nan_mask],
                                bins=200,
                                range=(-100, 100))[0]

        data_y = data[:, 1, :, :]
        nan_mask = data_y != data_y
        non_nan_mask = ~nan_mask
        histo_y_ = np.histogram(data_y[non_nan_mask],
                                bins=200,
                                range=(-100, 100))[0]

        data_z = data[:, 2, :, :]
        nan_mask = data_z != data_z
        non_nan_mask = ~nan_mask
        histo_z_ = np.histogram(data_z[non_nan_mask],
                                bins=200,
                                range=(-100, 100))[0]
        histo_x = histo_x + histo_x_
        histo_y = histo_y + histo_y_
        histo_z = histo_z + histo_z_

    print(' %8.4f %8.4f' %
          (np.min(np.array(min_values)), np.max(np.array(max_values))))
    csv_file = open('normals_%s.csv' % split_name, 'w')
    csv_file.write('histo_x, histo_y, histo_z\n')
    for i in range(200):
        csv_file.write('%d,%d, %d,\n' % (histo_x[i], histo_y[i], histo_z[i]))
    csv_file.close()


def check_depth_range(split_name):
    dataset = DepthDataset(DATASET_PATH, SPLITS_CSV_PATH, split_name)
    dataloader = DataLoader(dataset,
                            batch_size=100,
                            shuffle=False,
                            num_workers=20)
    print('split %s - %d' % (split_name, dataset.__len__()))
    histo = np.zeros(800)
    min_values = []
    max_values = []
    for batch in tqdm(dataloader):
        data, paths = batch
        nan_mask = data != data
        non_nan_mask = ~nan_mask
        histo_, bins = np.histogram(data[non_nan_mask],
                                    bins=800,
                                    range=(0, 800))
        histo = histo + histo_
        min_values.append(torch.min(data[non_nan_mask]))
        max_values.append(torch.max(data[non_nan_mask]))
    print(' %8.4f %8.4f' %
          (np.min(np.array(min_values)), np.max(np.array(max_values))))
    csv_file = open('depth_%s.csv' % split_name, 'w')
    csv_file.write('histo\n')
    for i in range(800):
        csv_file.write('%d,\n' % histo[i])
    csv_file.close()


if __name__ == "__main__":
    '''
    check_normals_range('train1')
    check_normals_range('train2')
    check_normals_range('train3')
    check_normals_range('valid')
    check_normals_range('test')
    '''
    status, status_code = check_arguments_without_delete(sys.argv)
    if status == 0:
        sys.exit(status_code + '\n' + usage_str)

    if RUN_TYPE == 0:
        get_gt_domains()
    else:
        get_exp_results()
