import shutil 
import os 
import sys
import numpy as np 
import cv2 
import torch 
import time
from tqdm import tqdm
import torch.utils.data as data 

import experts.raft_of_expert 
import experts.liteflownet_of_expert
import experts.sseg_fcn_expert
import experts.sseg_deeplabv3_expert
import experts.vmos_stm_expert

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
#    enable_double_check    [0/1] - if enabled, in case one expert folder already exists, 
#                                        the user will need to aknowledge if he wants to proceed or not 
#                                 - should not be set to 1 if the process runs in background 
#    working_h              [nr>0 / 0] - desired image height
#    working_w              [nr>0 / 0] - desired image width
#    expi                   - name of the i'th expert 
#                           - should be one of the VALID_EXPERTS_NAME

VALID_EXPERTS_NAME = ['of_fwd_raft', 'of_bwd_raft', 'of_fwd_liteflownet', 'of_bwd_liteflownet', 'sseg_fcn', 'sseg_deeplabv3', 'vmos_stm']
INPUT_PATH = r'/root/test_videos'
OUTPUT_PATH = r'/root/experts'
ENABLE_DOUBLE_CHECK = 1
EXPERTS_NAME = []
WORKING_H = 256
WORKING_W = 256

def check_arguments_and_init_paths(argv):
    global INPUT_PATH
    global OUTPUT_PATH
    global ENABLE_DOUBLE_CHECK
    global EXPERTS_NAME
    global WORKING_H
    global WORKING_W
    
    if len(argv)<7:
        status=0
        status_code = 'Incorrect usage'
        return status, status_code

    if not argv[1]=='-':
        INPUT_PATH = argv[1]
    if not argv[2]=='-':
        OUTPUT_PATH = argv[2]
    ENABLE_DOUBLE_CHECK = np.int32(argv[3])
    if not argv[4]=='0':
        WORKING_H = np.int32(argv[4])
    if not argv[5]=='0':
        WORKING_W = np.int32(argv[5])

    if not os.path.exists(INPUT_PATH) or len(os.listdir(INPUT_PATH))==0:
        status = 0
        status_code = 'Invalid input path'
        return status, status_code

    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    
    potential_experts = argv[6:]
    if argv[6]=='all':
        potential_experts = VALID_EXPERTS_NAME

    n_experts = len(potential_experts)#n_experts = len(argv) - 6

    EXPERTS_NAME = []
    for i in range(n_experts):
        exp_name = potential_experts[i]
        if not exp_name in VALID_EXPERTS_NAME:
            status = 0
            status_code = 'Expert %s is not valid'%exp_name
            return status, status_code
        exp_out_path = os.path.join(OUTPUT_PATH, exp_name)
        if ENABLE_DOUBLE_CHECK==1 and os.path.exists(exp_out_path) and len(os.listdir(exp_out_path)) > 0:
            value = input('Expert %s already exists. Proceed with deleating previous info?[y/n]'%exp_name)
            if value=='y':
                EXPERTS_NAME.append(exp_name)
        else:
            EXPERTS_NAME.append(exp_name)

    if len(EXPERTS_NAME) == 0:
        status = 0
        status_code = 'No experts remaining'

    for exp_name in EXPERTS_NAME:
        exp_out_path = os.path.join(OUTPUT_PATH, exp_name)
        if os.path.exists(exp_out_path):
            shutil.rmtree(exp_out_path)
        os.mkdir(exp_out_path)

    status = 1
    status_code = ''
    return status, status_code

def get_rgb_video_frames(vid_in_path):
    filenames = os.listdir(vid_in_path)
    filenames.sort()
    frames = []
    for filename in filenames:
        img = cv2.imread(os.path.join(vid_in_path, filename))
        img = cv2.resize(img, (WORKING_W, WORKING_H), cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    return frames

def get_expert(exp_name):
    if exp_name=='of_fwd_raft':
        return experts.raft_of_expert.RaftTest('experts/raft_optical_flow/models/raft-kitti.pth', 1)
    elif exp_name=='of_bwd_raft':
        return experts.raft_of_expert.RaftTest('experts/raft_optical_flow/models/raft-kitti.pth', 0)
    elif exp_name=='of_fwd_liteflownet':
        return experts.liteflownet_of_expert.LiteFlowNetTest('experts/liteflownet_optical_flow/models/liteflownet-default', 1)
    elif exp_name=='of_bwd_liteflownet':
        return experts.liteflownet_of_expert.LiteFlowNetTest('experts/liteflownet_optical_flow/models/liteflownet-default', 0)
    elif exp_name=='sseg_fcn':
        return experts.sseg_fcn_expert.FCNTest()
    elif exp_name=='sseg_deeplabv3':
        return experts.sseg_deeplabv3_expert.DeepLabv3Test()
    elif exp_name=='vmos_stm':
        return experts.vmos_stm_expert.STMTest('experts/vmos_stm/STM_weights.pth', 0, 21) 

def process_videos():
    videos_name = os.listdir(INPUT_PATH)
    videos_name.sort()
    n_videos = len(videos_name)
    for video_name in videos_name:
        vid_in_path = os.path.join(INPUT_PATH, video_name)
        frames = get_rgb_video_frames(vid_in_path)

        for exp_name in EXPERTS_NAME:
            expert = get_expert(exp_name)
            results = expert.apply_per_video(frames)

            vid_out_path = os.path.join(OUTPUT_PATH, exp_name, video_name)
            os.mkdir(vid_out_path)
            for i in range(len(results)):
                np.save(os.path.join(vid_out_path, '%08d.npy'%i), results[i])

if __name__ == "__main__":
    status, status_code = check_arguments_and_init_paths(sys.argv)
    if status == 0 :
        sys.exit(status_code +'\n'+ usage_str)
    
    process_videos()
    #value = input("do you want to continue? [y/n]")
    #import pdb 
    #pdb.set_trace()