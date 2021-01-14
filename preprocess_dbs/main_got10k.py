import glob
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

# if -1 => will process all movies
n_proc_videos = -1  #3000  #-1  #180  #3000
# if -1 => will process all frames
# if 0 => sample 4 frames per video
n_proc_frames_per_video = 0
# number of frames per video when n_proc_frameS_per_video==0
n_random_frames_per_video = 4

main_db_path = r'/data/tracking-vot/GOT-10k/train'
#main_exp_out_path = r'/data/multi-domain-graph/datasets/datasets_preproc_exp/GOT-10k/train'
main_exp_out_path = r'/data/multi-domain-graph-2/datasets/datasets_preproc_exp/GOT-10k_samples/train'
'''
main_db_path = r'/data/tracking-vot/GOT-10k/val'
#main_exp_out_path = r'/data/multi-domain-graph/datasets/datasets_preproc_exp/GOT-10k/val'
main_exp_out_path = r'/data/multi-domain-graph-2/datasets/datasets_preproc_exp/GOT-10k_samples/val'
'''

# to be ignored
'''
VALID_EXPERTS_NAME = [\
    'rgb', train -- mdg4
    'halftone_gray_basic',  -- train done
    'depth_sgdepth']   --train done 
'''
'''
VALID_EXPERTS_NAME = [\
    'edges_dexined',   -- train - mdg2 
    'normals_xtc',   - will run on mdg2 - when reaching this, stop and restart 
    'saliency_seg_egnet'] -- train - mdg3
'''
# up until here

VALID_EXPERTS_NAME = [\
    'sseg_fcn',
    'sseg_deeplabv3',
    'halftone_gray_basic', 'halftone_rgb_basic', 'halftone_cmyk_basic', 'halftone_rot_gray_basic',
    'depth_sgdepth',
    'depth_xtc',
    'edges_dexined',
    'normals_xtc',
    'saliency_seg_egnet',
    'rgb',
    'of_fwd_raft', 'of_bwd_raft',
    'of_fwd_liteflownet', 'of_bwd_liteflownet']

EXPERTS_NAME = []
CHECK_PREV_DATA = 0

usage_str = 'usage: python main_got10k.py check_prev_data exp1 exp2 ...'
#   check_prev_data         - [0/1] - 1 check prev data & ask if to delete; 0 otherwise
#    expi                   - name of the i'th expert / domain
#                           - should be one of the VALID_EXPERTS_NAME / VALID_GT_DOMAINS
#                           - 'all' to run all available experts / domains


def check_arguments(argv):
    global EXPERTS_NAME
    global CHECK_PREV_DATA

    if len(argv) < 3:
        return 0, 'incorrect usage'

    CHECK_PREV_DATA = np.int32(argv[1])

    if argv[2] == 'all':
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
        potential_experts = argv[2:]
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
        return experts.normals_expert.SurfaceNormalsModel(full_expert=True)
    elif exp_name == 'saliency_seg_egnet':
        return experts.saliency_seg_expert.SaliencySegmModel(full_expert=True)
    elif exp_name == 'rgb':
        return experts.rgb_expert.RGBModel(full_expert=True)


def get_rgb_video_frames(vid_in_path):

    filepaths = glob.glob(os.path.join(vid_in_path, '*.jpg'))
    filepaths.sort()
    frames = []
    out_filenames = []
    for filepath in filepaths:
        img = cv2.imread(filepath)
        img = cv2.resize(img, (WORKING_W, WORKING_H), cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
        base, tail = os.path.split(filepath)
        out_filenames.append(tail.replace('.jpg', '.npy'))
    return frames, out_filenames


class Dataset_ImgLevel(Dataset):
    def __init__(self, vid_in_path):
        super(Dataset_ImgLevel, self).__init__()

        filepaths = glob.glob(os.path.join(vid_in_path, '*.jpg'))
        filepaths.sort()
        if n_proc_frames_per_video == 0:
            indexes = np.arange(1, len(filepaths) - 1)
            np.random.seed(0)
            np.random.shuffle(indexes)
            indexes = indexes[0:n_random_frames_per_video]
            indexes = np.sort(
                np.concatenate((indexes - 1, indexes, indexes + 1)))
        else:
            indexes = np.arange(
                0,
                len(filepaths)
                if n_proc_frames_per_video == -1 else n_proc_frames_per_video)

        filepaths = [filepaths[i] for i in indexes]

        self.rgbs_path = []
        self.orig_indexes = indexes
        for filepath in filepaths:
            self.rgbs_path.append(os.path.join(vid_in_path, filepath))

    def __getitem__(self, index):
        rgb = get_image(self.rgbs_path[index])
        return rgb, index, self.orig_indexes[index]

    def __len__(self):
        return len(self.rgbs_path)


def get_exp_results():
    # TODO - change when using full videos

    videos_list_path = os.path.join(main_db_path, 'list.txt')
    file = open(videos_list_path, 'r')
    videos_name = file.readlines()
    videos_name.sort()
    file.close()
    videos_name = [video_name.strip() for video_name in videos_name]
    videos_name = videos_name[0:len(videos_name) if n_proc_videos ==
                              -1 else n_proc_videos]
    batch_size = 30
    video_dataloaders = []
    video_flow_dataloaders = []
    for video_name in videos_name:
        vid_in_path = os.path.join(main_db_path, video_name)
        dataset = Dataset_ImgLevel(vid_in_path)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 drop_last=False,
                                                 num_workers=8)
        video_dataloaders.append(dataloader)
        if n_proc_frames_per_video == 0:
            dataloader_flow = torch.utils.data.DataLoader(dataset,
                                                          batch_size=3,
                                                          shuffle=False,
                                                          drop_last=False,
                                                          num_workers=8)
            video_dataloaders.append(dataloader_flow)
        else:
            video_flow_dataloaders.append(dataloader)

    for exp_name in EXPERTS_NAME:
        print("EXPERT: %20s" % exp_name)
        expert = get_expert(exp_name)
        if 'of_' in exp_name:
            local_video_dataloaders = video_flow_dataloaders
        else:
            local_video_dataloaders = video_dataloaders

        for video_data in zip(videos_name, local_video_dataloaders):
            video_name, dataloader = video_data

            vid_out_path = os.path.join(main_exp_out_path, exp_name,
                                        video_name)
            os.makedirs(vid_out_path)

            for batch_idx, (frames, indexes,
                            orig_indexes) in enumerate(tqdm(dataloader)):
                results = expert.apply_expert_batch(frames)

                for sample in zip(results, indexes, orig_indexes):
                    expert_res, sample_idx, orig_sample_idx = sample
                    out_path = os.path.join(
                        vid_out_path,
                        '%08d_%08d.npy' % (sample_idx, orig_sample_idx))
                    np.save(out_path, expert_res)


if __name__ == "__main__":
    status, status_code = check_arguments(sys.argv)
    if status == 0:
        sys.exit(status_code + '\n' + usage_str)

    get_exp_results()
