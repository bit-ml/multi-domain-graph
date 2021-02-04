import os
import sys
import traceback

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

main_db_path = r'/data/multi-domain-graph-6/datasets/replica_raw/test'
main_gt_out_path = r'/data/multi-domain-graph-6/datasets/datasets_preproc_gt/replica/test'
main_exp_out_path = r'/data/multi-domain-graph-6/datasets/datasets_preproc_exp/replica/test'

# main_db_path = r'/data/multi-domain-graph-6/datasets/replica_raw/train'
# main_gt_out_path = r'/data/multi-domain-graph-6/datasets/datasets_preproc_gt/replica/train'
# main_exp_out_path = r'/data/multi-domain-graph-6/datasets/datasets_preproc_exp/replica/train'

# main_db_path = r'/data/multi-domain-graph-6/datasets/replica_raw/val'
# main_gt_out_path = r'/data/multi-domain-graph-6/datasets/datasets_preproc_gt/replica/val'
# main_exp_out_path = r'/data/multi-domain-graph-6/datasets/datasets_preproc_exp/replica/val'

# dataset domain names
VALID_ORIG_GT_DOMAINS = ['rgb', 'depth', 'normals']

# our internal domain names
VALID_GT_DOMAINS = ['rgb', 'depth', 'normals']

VALID_EXPERTS_NAME = [
    'depth_xtc',
    'normals_xtc',
    'edges_dexined',
    'halftone_gray_basic',
    'saliency_seg_egnet',
]

RUN_TYPE = []
EXPERTS_NAME = []
ORIG_DOMAINS = []
DOMAINS = []

usage_str = 'usage: python main_taskonomy.py type exp1 exp2 ...'
#    type                   - [0/1] - 0 create preprocessed gt samples
#                                   - 1 create preprocessed experts samples
#    expi                   - name of the i'th expert / domain
#                           - should be one of the VALID_EXPERTS_NAME / VALID_GT_DOMAINS
#                           - 'all' to run all available experts / domains


def check_arguments_without_delete(argv):
    global RUN_TYPE
    global EXPERTS_NAME
    global ORIG_DOMAINS
    global DOMAINS

    if len(argv) < 3:
        return 0, 'incorrect usage'

    RUN_TYPE = np.int32(argv[1])
    if not (RUN_TYPE == 0 or RUN_TYPE == 1 or RUN_TYPE == 2 or RUN_TYPE == 3):
        return 0, 'incorrect run type: %d' % RUN_TYPE

    if RUN_TYPE == 0:
        if argv[2] == 'all':
            ORIG_DOMAINS = []
            DOMAINS = []
            for doms in zip(VALID_ORIG_GT_DOMAINS, VALID_GT_DOMAINS):
                orig_dom_name, dom_name = doms
                dom_out_path = os.path.join(main_gt_out_path, dom_name)
                ORIG_DOMAINS.append(orig_dom_name)
                DOMAINS.append(dom_name)
        else:
            potential_domains = argv[2:]
            print("potential_domains", potential_domains)
            print("VALID_GT_DOMAINS", VALID_GT_DOMAINS)
            ORIG_DOMAINS = []
            DOMAINS = []
            for i in range(len(potential_domains)):
                dom_name = potential_domains[i]
                if not dom_name in VALID_GT_DOMAINS:
                    status = 0
                    status_code = 'Domain %s is not valid' % dom_name
                    return status, status_code
                orig_dom_name = VALID_ORIG_GT_DOMAINS[VALID_GT_DOMAINS.index(
                    dom_name)]
                dom_out_path = os.path.join(main_gt_out_path, dom_name)

                ORIG_DOMAINS.append(orig_dom_name)
                DOMAINS.append(dom_name)
        print("ORIG_DOMAINS", ORIG_DOMAINS)
        return 1, ''
    elif RUN_TYPE == 1:
        if argv[2] == 'all':
            EXPERTS_NAME = []
            for exp_name in VALID_EXPERTS_NAME:
                exp_out_path = os.path.join(main_exp_out_path, exp_name)
                EXPERTS_NAME.append(exp_name)
        else:
            potential_experts = argv[2:]
            EXPERTS_NAME = []
            for i in range(len(potential_experts)):
                exp_name = potential_experts[i]
                if not exp_name in VALID_EXPERTS_NAME:
                    status = 0
                    status_code = 'Expert %s is not valid' % exp_name
                    return status, status_code
                exp_out_path = os.path.join(main_exp_out_path, exp_name)
                EXPERTS_NAME.append(exp_name)
        return 1, ''
    else:
        return 1, ''


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


def process_rgb(in_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    os.system("cp -r %s/* %s" % (in_path, out_path))


def process_depth(in_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    files = os.listdir(in_path)
    files.sort()

    for file_ in files:
        depth_map = np.load(os.path.join(in_path, file_))
        depth_map = depth_map / 14
        depth_map = 1 - depth_map
        depth_map = depth_map[None]
        np.save(os.path.join(out_path, file_), depth_map)


def process_surface_normals(in_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    os.system("cp %s/* %s/" % (in_path, out_path))


def get_gt_domains():
    print("get_gt_domains", ORIG_DOMAINS)
    for doms in zip(ORIG_DOMAINS, DOMAINS):
        orig_dom_name, dom_name = doms

        in_path = os.path.join(main_db_path, orig_dom_name)
        out_path = os.path.join(main_gt_out_path, dom_name)
        if orig_dom_name == 'rgb':
            process_rgb(in_path, out_path)
        elif orig_dom_name == 'depth':
            process_depth(in_path, out_path)
        elif orig_dom_name == 'normals':
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
        # try:
        rgb = np.load(self.rgbs_path[index])
        # except:
        #     traceback.print_exc()
        # print("ERROARE::: path", self.rgbs_path[index], "index", index)
        # rgb = np.zeros((WORKING_W, WORKING_H, 3), dtype=np.float32)
        index_of_file = int(self.rgbs_path[index].split("/")[-1].replace(
            ".npy", ""))
        return rgb, index_of_file

    def __len__(self):
        return len(self.rgbs_path)


def get_exp_results():
    with torch.no_grad():
        rgbs_path = os.path.join(main_db_path, 'rgb')
        batch_size = 100
        dataset = Dataset_ImgLevel(rgbs_path)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 drop_last=False,
                                                 num_workers=0)

        for exp_name in EXPERTS_NAME:
            print('EXPERT: %20s' % exp_name)
            expert = get_expert(exp_name)

            exp_out_path = os.path.join(main_exp_out_path, exp_name)
            os.makedirs(exp_out_path, exist_ok=True)

            for batch_idx, (frames, indexes) in enumerate(tqdm(dataloader)):
                # skip fast (eg. for missing depth 00161500.npy)
                # if indexes[-1] < 161499:
                #     continue
                # already_exists = 0
                # for check_idx in indexes:
                #     out_path = os.path.join(exp_out_path, '%08d.npy' % check_idx)
                #     if os.path.exists(out_path):
                #         already_exists += 1
                # if already_exists == len(indexes):
                #     continue

                frames = frames.permute(0, 2, 3, 1) * 255.
                results = expert.apply_expert_batch(frames)

                for sample in zip(results, indexes):
                    expert_res, sample_idx = sample

                    out_path = os.path.join(exp_out_path,
                                            '%08d.npy' % sample_idx)
                    # if os.path.exists(out_path):
                    #     continue

                    np.save(out_path, expert_res)


def split_train_gt_ds():
    all_domains = VALID_GT_DOMAINS

    for domain in all_domains:
        dom_in_path = os.path.join(main_gt_out_path, domain)

        dom_out_path1 = os.path.join(main_gt_out_path_1, domain)
        dom_out_path2 = os.path.join(main_gt_out_path_2, domain)
        os.makedirs(dom_out_path1, exist_ok=True)
        os.makedirs(dom_out_path2, exist_ok=True)

        files = os.listdir(dom_in_path)
        files.sort()
        files1 = files[0:4800]
        files2 = files[4800:]

        for file_ in files1:
            src_path = os.path.join(dom_in_path, file_)
            dst_path = os.path.join(dom_out_path1, file_)
            os.system("cp -r %s %s" % (src_path, dst_path))
        for file_ in files2:
            src_path = os.path.join(dom_in_path, file_)
            dst_path = os.path.join(dom_out_path2, file_)
            os.system("cp -r %s %s" % (src_path, dst_path))


def split_train_exp_ds():
    all_domains = VALID_EXPERTS_NAME

    for domain in all_domains:
        dom_in_path = os.path.join(main_exp_out_path, domain)

        dom_out_path1 = os.path.join(main_exp_out_path_1, domain)
        dom_out_path2 = os.path.join(main_exp_out_path_2, domain)
        os.makedirs(dom_out_path1, exist_ok=True)
        os.makedirs(dom_out_path2, exist_ok=True)

        files = os.listdir(dom_in_path)
        files.sort()
        files1 = files[0:4800]
        files2 = files[4800:]

        for file_ in files1:
            src_path = os.path.join(dom_in_path, file_)
            dst_path = os.path.join(dom_out_path1, file_)
            os.system("cp -r %s %s" % (src_path, dst_path))
        for file_ in files2:
            src_path = os.path.join(dom_in_path, file_)
            dst_path = os.path.join(dom_out_path2, file_)
            os.system("cp -r %s %s" % (src_path, dst_path))


if __name__ == "__main__":
    status, status_code = check_arguments_without_delete(sys.argv)
    if status == 0:
        sys.exit(status_code + '\n' + usage_str)

    if RUN_TYPE == 0:
        get_gt_domains()
    elif RUN_TYPE == 1:
        get_exp_results()
    elif RUN_TYPE == 2:
        split_train_gt_ds()
    else:
        split_train_exp_ds()
