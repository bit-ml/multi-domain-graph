import shutil
import os
import sys
import numpy as np
import cv2
import torch
import time
from tqdm import tqdm
import torch.utils.data as data
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import experts.raft_of_expert
import experts.liteflownet_of_expert
import experts.sseg_fcn_expert
import experts.sseg_deeplabv3_expert
import experts.vmos_stm_expert
import experts.halftone_expert

import utils.visualize_of
import utils.visualize_vmos

usage_str = 'usage: python main_save_experts.py input_path output_path enable_double_check working_h working_w exp1 exp2 ...'
# usage
# python main_save_experts path1 path2 enable_double_check exp1 exp2
#    path1                  [a path / '-'] - videos input path
#                                           - if '-' is provided, we use INPUT_PATH's default value
#    path2                  [a path / '-'] - experts main path
#                                           - if '-' is provided, we use OUTPUT_PATH's default value
#                           - info of one expert will be saved on path2/exp_name
#    db_name                dataset name; if '' it will be ignored
#                                       - if '-' is provided, we use DB_NAME's default value
#    subset_name            subset name; if '' it will be ignored
#                                       - if '-' is provided, we use SUBSET_NAME's default value
#    enable_double_check    [0/1] - if enabled, in case one expert folder already exists,
#                                        the user will need to aknowledge if he wants to proceed or not
#                                 - should not be set to 1 if the process runs in background
#    working_h              [nr>0 / 0] - desired image height
#    working_w              [nr>0 / 0] - desired image width
#    expi                   - name of the i'th expert
#                           - should be one of the VALID_EXPERTS_NAME

VALID_EXPERTS_NAME = [\
    'of_fwd_raft', 'of_bwd_raft',
    'of_fwd_liteflownet', 'of_bwd_liteflownet',
    'sseg_fcn',
    'sseg_deeplabv3',
    'vmos_stm',
    'halftone_gray_basic', 'halftone_rgb_basic', 'halftone_cmyk_basic', 'halftone_rot_gray_basic']

INPUT_PATH = r'/data/tracking-vot/GOT-10k/train'
OUTPUT_PATH = r'/data/experts-output'
ENABLE_DOUBLE_CHECK = 1
EXPERTS_NAME = []
WORKING_H = 256
WORKING_W = 256
DB_NAME = 'GOT-10k'
SUBSET_NAME = 'train'


def check_arguments_and_init_paths(argv):
    global INPUT_PATH
    global OUTPUT_PATH
    global ENABLE_DOUBLE_CHECK
    global EXPERTS_NAME
    global WORKING_H
    global WORKING_W
    global DB_NAME
    global SUBSET_NAME

    if len(argv) < 9:
        status = 0
        status_code = 'Incorrect usage'
        return status, status_code

    if not argv[1] == '-':
        INPUT_PATH = argv[1]
    if not argv[2] == '-':
        OUTPUT_PATH = argv[2]
    if not argv[3] == '-':
        DB_NAME = argv[3]
    if not argv[4] == '-':
        SUBSET_NAME = argv[4]
    ENABLE_DOUBLE_CHECK = np.int32(argv[5])
    if not argv[6] == '0':
        WORKING_H = np.int32(argv[6])
    if not argv[7] == '0':
        WORKING_W = np.int32(argv[7])

    if not os.path.exists(INPUT_PATH) or len(os.listdir(INPUT_PATH)) == 0:
        status = 0
        status_code = 'Invalid input path'
        return status, status_code

    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    potential_experts = argv[8:]
    if argv[8] == 'all':
        potential_experts = VALID_EXPERTS_NAME

    n_experts = len(potential_experts)  #n_experts = len(argv) - 6

    EXPERTS_NAME = []
    for i in range(n_experts):
        exp_name = potential_experts[i]
        if not exp_name in VALID_EXPERTS_NAME:
            status = 0
            status_code = 'Expert %s is not valid' % exp_name
            return status, status_code
        exp_out_path = os.path.join(OUTPUT_PATH, exp_name)

        if not DB_NAME == '':
            exp_out_path = os.path.join(exp_out_path, DB_NAME)
        if not SUBSET_NAME == '':
            exp_out_path = os.path.join(exp_out_path, SUBSET_NAME)

        if ENABLE_DOUBLE_CHECK == 1 and os.path.exists(exp_out_path) and len(
                os.listdir(exp_out_path)) > 0:
            value = input(
                'Expert %s already exists. Proceed with deleating previous info (%s)?[y/n]'
                % (exp_name, exp_out_path))
            if value == 'y':
                EXPERTS_NAME.append(exp_name)
        else:
            EXPERTS_NAME.append(exp_name)

    if len(EXPERTS_NAME) == 0:
        status = 0
        status_code = 'No experts remaining'

    for exp_name in EXPERTS_NAME:
        exp_out_path = os.path.join(OUTPUT_PATH, exp_name)
        if DB_NAME == '':
            if os.path.exists(exp_out_path):
                shutil.rmtree(exp_out_path)
            os.mkdir(exp_out_path)
        else:
            if not os.path.exists(exp_out_path):
                os.mkdir(exp_out_path)
            exp_out_path = os.path.join(exp_out_path, DB_NAME)
            if SUBSET_NAME == '':
                if os.path.exists(exp_out_path):
                    shutil.rmtree(exp_out_path)
                os.mkdir(exp_out_path)
            else:
                if not os.path.exists(exp_out_path):
                    os.mkdir(exp_out_path)
                exp_out_path = os.path.join(exp_out_path, SUBSET_NAME)
                if os.path.exists(exp_out_path):
                    shutil.rmtree(exp_out_path)
                os.mkdir(exp_out_path)

    status = 1
    status_code = ''
    return status, status_code


def get_rgb_video_frames(vid_in_path):
    import glob
    filepaths = glob.glob(os.path.join(vid_in_path, '*.jpg'))
    #filenames = os.listdir(vid_in_path)
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


def process_videos():
    videos_list_path = os.path.join(INPUT_PATH, 'list.txt')
    file = open(videos_list_path, 'r')
    videos_name = file.readlines()
    videos_name.sort()
    file.close()
    videos_name = [video_name.strip() for video_name in videos_name]
    videos_name = videos_name[0:3298]
    #videos_name = os.listdir(INPUT_PATH)
    #videos_name.sort()
    n_videos = len(videos_name)
    for video_name in videos_name:
        vid_in_path = os.path.join(INPUT_PATH, video_name)
        frames, out_filenames = get_rgb_video_frames(vid_in_path)

        n_frames = len(frames)
        n_frames = min(5, n_frames)
        frames = frames[0:n_frames]

        for exp_name in EXPERTS_NAME:
            expert = get_expert(exp_name)
            results = expert.apply_expert(frames)

            exp_out_path = os.path.join(OUTPUT_PATH, exp_name)
            if not DB_NAME == '':
                exp_out_path = os.path.join(exp_out_path, DB_NAME)
            if not SUBSET_NAME == '':
                exp_out_path = os.path.join(exp_out_path, SUBSET_NAME)
            vid_out_path = os.path.join(exp_out_path, video_name)
            os.mkdir(vid_out_path)
            #for i in range(len(results)):
            #    np.save(os.path.join(vid_out_path, out_filenames[i]), results[i])
            for i in range(3):
                np.save(os.path.join(vid_out_path, out_filenames[i]),
                        results[i])


if __name__ == "__main__":
    status, status_code = check_arguments_and_init_paths(sys.argv)
    if status == 0:
        sys.exit(status_code + '\n' + usage_str)

    process_videos()
    #value = input("do you want to continue? [y/n]")
    #import pdb
    #pdb.set_trace()