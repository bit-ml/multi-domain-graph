import os
import shutil
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.insert(0,
                os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import experts.depth_expert
import experts.edges_expert
import experts.halftone_expert
import experts.liteflownet_of_expert
import experts.normals_expert
import experts.raft_of_expert
import experts.rgb_expert
import experts.saliency_seg_expert
import experts.sseg_deeplabv3_expert
import experts.sseg_fcn_expert
import experts.vmos_stm_expert

WORKING_H = 256
WORKING_W = 256

# main_db_path = r'/data/multi-domain-graph/datasets/taskonomy/taskonomy-sample-model-1-master'
# main_gt_out_path = r'/data/multi-domain-graph/datasets/datasets_preproc_gt/taskonomy/sample-model'
# main_exp_out_path = r'/data/multi-domain-graph/datasets/datasets_preproc_exp/taskonomy/sample-model'

main_db_path = r'/data/multi-domain-graph-3/datasets/Taskonomy/tiny-test'
main_gt_out_path = r'/data/multi-domain-graph-3/datasets/datasets_preproc_gt/taskonomy/tiny-test'
main_exp_out_path = r'/data/multi-domain-graph-3/datasets/datasets_preproc_exp/taskonomy/tiny-test'

# dataset domain names
VALID_ORIG_GT_DOMAINS = ['rgb', 'depth_zbuffer', 'edge_texture', 'normal']

# our internal domain names
VALID_GT_DOMAINS = ['rgb', 'depth', 'edges', 'normals']

VALID_EXPERTS_NAME = [\
    'prdimp50',
    'sseg_fcn',
    'sseg_deeplabv3',
    'halftone_gray_basic', 'halftone_rgb_basic', 'halftone_cmyk_basic', 'halftone_rot_gray_basic',
    'depth_sgdepth',
    'depth_xtc',
    'edges_dexined',
    'normals_xtc',
    'saliency_seg_egnet',
    'rgb']
'''
VALID_EXPERTS_NAME = [\
    'halftone_gray_basic', 'halftone_rgb_basic',
    'depth_sgdepth',
    'edges_dexined',
    'sseg_fcn']
'''
'''
VALID_EXPERTS_NAME = [\
    'halftone_cmyk_basic', 'halftone_rot_gray_basic',
    'normals_xtc',
    'saliency_seg_egnet',
    'rgb',
    'sseg_deeplabv3']
'''
RUN_TYPE = []
EXPERTS_NAME = []
ORIG_DOMAINS = []
DOMAINS = []
CHECK_PREV_DATA = 0

usage_str = 'usage: python main_taskonomy.py type check_prev_data exp1 exp2 ...'
#    type                   - [0/1] - 0 create preprocessed gt samples
#                                   - 1 create preprocessed experts samples
#   check_prev_data         - [0/1] - 1 check prev data & ask if to delete; 0 otherwise
#    expi                   - name of the i'th expert / domain
#                           - should be one of the VALID_EXPERTS_NAME / VALID_GT_DOMAINS
#                           - 'all' to run all available experts / domains


def check_arguments(argv):
    global RUN_TYPE
    global EXPERTS_NAME
    global ORIG_DOMAINS
    global DOMAINS
    global CHECK_PREV_DATA

    if len(argv) < 4:
        return 0, 'incorrect usage'

    RUN_TYPE = np.int32(argv[1])
    if not (RUN_TYPE == 0 or RUN_TYPE == 1):
        return 0, 'incorrect run type: %d' % RUN_TYPE
    CHECK_PREV_DATA = np.int32(argv[2])

    if RUN_TYPE == 0:
        if argv[3] == 'all':
            ORIG_DOMAINS = []
            DOMAINS = []
            for doms in zip(VALID_ORIG_GT_DOMAINS, VALID_GT_DOMAINS):
                orig_dom_name, dom_name = doms
                dom_out_path = os.path.join(main_gt_out_path, dom_name)
                if CHECK_PREV_DATA and os.path.exists(dom_out_path):
                    value = input(
                        'Domain %s already exists. Proceed with deleating previous info (%s)?[y/n]'
                        % (dom_name, dom_out_path))
                    if value == 'y':
                        shutil.rmtree(dom_out_path)
                        ORIG_DOMAINS.append(orig_dom_name)
                        DOMAINS.append(dom_name)
                elif not CHECK_PREV_DATA and os.path.exists(dom_out_path):
                    shutil.rmtree(dom_out_path)
                else:
                    ORIG_DOMAINS.append(orig_dom_name)
                    DOMAINS.append(dom_name)
        else:
            potential_domains = argv[3:]
            ORIG_DOMAINS = []
            DOMAINS = []
            for i in range(len(potential_domains)):
                dom_name = potential_domains[i]
                if not dom_name in VALID_GT_DOMAINS:
                    status = 0
                    status_code = 'Domain %s is not valid' % orig_dom_name
                    return status, status_code
                orig_dom_name = VALID_ORIG_GT_DOMAINS[VALID_GT_DOMAINS.index(
                    dom_name)]
                dom_out_path = os.path.join(main_gt_out_path, dom_name)
                if CHECK_PREV_DATA and os.path.exists(dom_out_path):
                    value = input(
                        'Domain %s already exists. Proceed with deleating previous info (%s)?[y/n]'
                        % (dom_name, dom_out_path))
                    if value == 'y':
                        shutil.rmtree(dom_out_path)
                        ORIG_DOMAINS.append(orig_dom_name)
                        DOMAINS.append(dom_name)
                elif not CHECK_PREV_DATA and os.path.exists(dom_out_path):
                    shutil.rmtree(dom_out_path)
                else:
                    ORIG_DOMAINS.append(orig_dom_name)
                    DOMAINS.append(dom_name)
        return 1, ''
    else:
        if argv[3] == 'all':
            EXPERTS_NAME = []
            for exp_name in VALID_EXPERTS_NAME:
                exp_out_path = os.path.join(main_exp_out_path, exp_name)
                if CHECK_PREV_DATA and os.path.exists(exp_out_path):
                    value = input(
                        'Domain %s already exists. Proceed with deleating previous info (%s)?[y/n]'
                        % (exp_name, exp_out_path))
                    if value == 'y':
                        shutil.rmtree(exp_out_path)
                        EXPERTS_NAME.append(exp_name)
                elif not CHECK_PREV_DATA and os.path.exists(exp_out_path):
                    shutil.rmtree(exp_out_path)
                else:
                    EXPERTS_NAME.append(exp_name)
        else:
            potential_experts = argv[3:]
            EXPERTS_NAME = []
            for i in range(len(potential_experts)):
                exp_name = potential_experts[i]
                if not exp_name in VALID_EXPERTS_NAME:
                    status = 0
                    status_code = 'Expert %s is not valid' % exp_name
                    return status, status_code
                exp_out_path = os.path.join(main_exp_out_path, exp_name)
                if CHECK_PREV_DATA and os.path.exists(exp_out_path):
                    value = input(
                        'Domain %s already exists. Proceed with deleating previous info (%s)?[y/n]'
                        % (exp_name, exp_out_path))
                    if value == 'y':
                        shutil.rmtree(exp_out_path)
                        EXPERTS_NAME.append(exp_name)
                elif not CHECK_PREV_DATA and os.path.exists(exp_out_path):
                    shutil.rmtree(exp_out_path)
                else:
                    EXPERTS_NAME.append(exp_name)
        return 1, ''


def get_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (WORKING_W, WORKING_H), cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_pil_image(img_path):
    img = Image.open(img_path)
    img = img.resize((WORKING_H, WORKING_W))
    return img


def get_expert(exp_name):
    if exp_name == 'of_fwd_raft':
        return experts.raft_of_expert.RaftModel(full_expert=True, fwd=1)
    elif exp_name == 'of_bwd_raft':
        return experts.raft_of_expert.RaftModel(full_expert=True, fwd=0)
    elif exp_name == 'of_fwd_liteflownet':
        return experts.liteflownet_of_expert.LiteFlowNetModel(full_expert=True,
                                                              fwd=1)
    elif exp_name == 'of_bwd_liteflownet':
        return experts.liteflownet_of_expert.LiteFlowNetModel(full_expert=True,
                                                              fwd=0)
    elif exp_name == 'sseg_fcn':
        return experts.sseg_fcn_expert.FCNModel(full_expert=True)
    elif exp_name == 'sseg_deeplabv3':
        return experts.sseg_deeplabv3_expert.DeepLabv3Model(full_expert=True)
    elif exp_name == 'vmos_stm':
        return experts.vmos_stm_expert.STMModel(
            'experts/vmos_stm/STM_weights.pth', 0, 21)
    elif exp_name == 'halftone_gray_basic':
        return experts.halftone_expert.HalftoneModel(full_expert=True, style=0)
    elif exp_name == 'halftone_rgb_basic':
        return experts.halftone_expert.HalftoneModel(full_expert=True, style=1)
    elif exp_name == 'halftone_cmyk_basic':
        return experts.halftone_expert.HalftoneModel(full_expert=True, style=2)
    elif exp_name == 'halftone_rot_gray_basic':
        return experts.halftone_expert.HalftoneModel(full_expert=True, style=3)
    elif exp_name == 'depth_sgdepth':
        sys.argv = ['']
        return experts.depth_expert.DepthModel(full_expert=True)
    elif exp_name == 'depth_xtc':
        return experts.depth_expert.DepthModelXTC(full_expert=True)
    elif exp_name == 'edges_dexined':
        return experts.edges_expert.EdgesModel(full_expert=True)
    elif exp_name == 'normals_xtc':
        return experts.normals_expert.SurfaceNormalsXTC(full_expert=True)
    elif exp_name == 'saliency_seg_egnet':
        return experts.saliency_seg_expert.SaliencySegmModel(full_expert=True)
    elif exp_name == 'rgb':
        return experts.rgb_expert.RGBModel(full_expert=True)


def get_data_range(in_path):
    filenames = os.listdir(in_path)
    filenames.sort()
    min_values = []
    max_values = []
    for filename in filenames:
        data_path = os.path.join(in_path, filename)
        data = Image.open(data_path)
        data = np.array(data)
        min_values.append(np.min(data))
        max_values.append(np.max(data))
    min_value = np.min(np.array(min_values))
    max_values = np.array(max_values)
    max_value = np.max(max_values)
    max_values[max_values == 65535] = 0
    second_max_value = np.max(max_values)
    print('range: -- min: %20.10f max: %20.10f second_max: %20.10f' %
          (min_value, max_value, second_max_value))
    return min_value, max_value, second_max_value


def process_rgb(in_path, out_path):
    os.makedirs(out_path)
    filenames = os.listdir(in_path)
    filenames.sort()
    idx = 0
    for filename in filenames:
        img = get_image(os.path.join(in_path, filename))
        img = img.astype('float32')
        img = np.moveaxis(img, 2, 0) / 255
        out_img_path = os.path.join(out_path, '%08d.npy' % idx)
        np.save(out_img_path, img)
        idx = idx + 1


def process_depth(in_path, out_path):
    os.makedirs(out_path)
    filenames = os.listdir(in_path)
    filenames.sort()
    min_value, max_value, second_max_value = get_data_range(in_path)
    #min_value = 102
    #max_value = 65535
    #second_max_value = 7683
    idx = 0
    for filename in filenames:
        data_path = os.path.join(in_path, filename)
        data = Image.open(data_path)
        data = np.array(data)
        data = data.astype('float32')
        data = data[:, :, None]
        data = torch.tensor(data, dtype=torch.float32)
        data = data.permute(2, 0, 1)
        data = torch.nn.functional.interpolate(data[None],
                                               (WORKING_H, WORKING_W))[0]
        data[data == max_value] = (min_value + second_max_value) / 2
        data = (data - min_value) / second_max_value
        data = 1 - data
        data = data.numpy()
        out_img_path = os.path.join(out_path, '%08d.npy' % idx)
        np.save(out_img_path, data)
        idx = idx + 1


def process_edges(in_path, out_path):
    os.makedirs(out_path)
    filenames = os.listdir(in_path)
    filenames.sort()
    idx = 0
    _, max_value, _ = get_data_range(in_path)
    # max_value = 11355
    for filename in filenames:
        data_path = os.path.join(in_path, filename)
        data = Image.open(data_path)
        data = np.array(data)
        data = data.astype('float32')
        data = data[:, :, None]
        data = torch.tensor(data, dtype=torch.float32)
        data = data.permute(2, 0, 1)
        data = torch.nn.functional.interpolate(data[None],
                                               (WORKING_H, WORKING_W))[0]
        data = data / max_value
        data = data.numpy()
        out_img_path = os.path.join(out_path, '%08d.npy' % idx)
        np.save(out_img_path, data)
        idx = idx + 1


def process_surface_normals(in_path, out_path):
    os.makedirs(out_path)
    filenames = os.listdir(in_path)
    filenames.sort()
    _, max_value, _ = get_data_range(in_path)
    # min_value = 0
    # max_value = 255
    idx = 0
    for filename in filenames:
        data_path = os.path.join(in_path, filename)
        data = Image.open(data_path)
        data = np.array(data)
        data = data.astype('float32')
        data = torch.tensor(data, dtype=torch.float32)
        data = data.permute(2, 0, 1)
        data = torch.nn.functional.interpolate(data[None],
                                               (WORKING_H, WORKING_W))[0]
        data = data / max_value
        data = data.numpy()
        out_img_path = os.path.join(out_path, '%08d.npy' % idx)
        np.save(out_img_path, data)
        idx = idx + 1


def get_gt_domains():
    for doms in zip(ORIG_DOMAINS, DOMAINS):
        orig_dom_name, dom_name = doms

        in_path = os.path.join(main_db_path, orig_dom_name)
        out_path = os.path.join(main_gt_out_path, dom_name)

        if orig_dom_name == 'rgb':
            process_rgb(in_path, out_path)
        elif orig_dom_name == 'depth_zbuffer':
            process_depth(in_path, out_path)
        elif orig_dom_name == 'edge_texture':
            process_edges(in_path, out_path)
        elif orig_dom_name == 'normal':
            process_surface_normals(in_path, out_path)


class Dataset_ImgLevel(Dataset):
    def __init__(self, rgbs_path):
        super(Dataset_ImgLevel, self).__init__()
        filenames = os.listdir(rgbs_path)
        filenames.sort()
        self.rgbs_path = []
        for filename in filenames:
            self.rgbs_path.append(os.path.join(rgbs_path, filename))

    def __getitem__(self, index):
        rgb = get_image(self.rgbs_path[index])
        return rgb, index

    def __len__(self):
        return len(self.rgbs_path)


def get_exp_results():
    rgbs_path = os.path.join(main_db_path, 'rgb')
    batch_size = 25
    dataset = Dataset_ImgLevel(rgbs_path)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=8)

    for exp_name in EXPERTS_NAME:
        print('EXPERT: %20s' % exp_name)
        expert = get_expert(exp_name)
        exp_out_path = os.path.join(main_exp_out_path, exp_name)
        os.makedirs(exp_out_path)

        for batch_idx, (frames, indexes) in enumerate(tqdm(dataloader)):
            results = expert.apply_expert_batch(frames)

            for sample in zip(results, indexes):
                expert_res, sample_idx = sample
                out_path = os.path.join(exp_out_path, '%08d.npy' % sample_idx)
                np.save(out_path, expert_res)


if __name__ == "__main__":
    status, status_code = check_arguments(sys.argv)
    if status == 0:
        sys.exit(status_code + '\n' + usage_str)

    if RUN_TYPE == 0:
        get_gt_domains()
    else:
        get_exp_results()
