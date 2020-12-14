import shutil
import os
import sys
import cv2
import torch
import time
import numpy as np
import torch.utils.data as data
from PIL import Image

#print(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0,
                os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import experts.raft_of_expert
import experts.liteflownet_of_expert
import experts.sseg_fcn_expert
import experts.sseg_deeplabv3_expert
import experts.vmos_stm_expert
import experts.halftone_expert
import experts.depth_expert
import experts.edges_expert
import experts.normals_expert
import experts.saliency_seg_expert

# run experts for img dataset, store results and test them
#   - all experts will run at a working res (256x256), results will be saved in this res
#   - evaluation on both working res and original res

usage_str = 'usage: python main_img_test_experts.py type input_path output_path enable_double_check working_h working_w exp1 exp2 ...'
# python main_img_text_experts.py 0 - - 1 0 0 all
# usage
# python main_img_test_experts type path1 path2 enable_double_check exp1 exp2
#    type                   [0/1/2] - 0 - only run experts & save results
#                                   - 1 - only test results - not implemented
#                                   - 2 - run experts & test - not implemented
#    path1                  [a path / '-'] - imgs input path
#                                          - if '-' is provided, we use INPUT_PATH's default value
#    path2                  [a path / '-'] - experts main path
#                                          - if '-' is provided, we use OUTPUT_PATH's default value
#                           - info of one expert will be saved on path2/exp_name
#    enable_double_check    [0/1] - if enabled, in case one expert folder already exists,
#                                   the user will need to aknowledge if he wants to proceed or not
#                           - should not be set to 1 if the process runs in the background
#    working_h              [nr>0 / 0] - desired image height
#    working_w              [nr>0 / 0] - desired image width
#    expi                   - name of the i'th expert
#                           - should be one of the VALID_EXPERTS_NAME

INPUT_PATH = r'/data/multi-domain-graph/datasets/taskonomy/taskonomy-sample-model-1-master/rgb'
OUTPUT_PATH = r'/data/multi-domain-graph/datasets/taskonomy/taskonomy-sample-model-1-master-experts'
ENABLE_DOUBLE_CHECK = 1
EXPERTS_NAME = []
WORKING_H = 256
WORKING_W = 256
RUN_TYPE = 0

VALID_EXPERTS_NAME = [\
    'sseg_fcn',
    'sseg_deeplabv3',
    'halftone_gray_basic', 'halftone_rgb_basic', 'halftone_cmyk_basic', 'halftone_rot_gray_basic',
    'depth_sgdepth',
    'edges_dexined',
    'normals_xtc',
    'saliency_seg_egnet',
    'rgb']


def check_arguments_and_init_paths(argv):
    global INPUT_PATH
    global OUTPUT_PATH
    global ENABLE_DOUBLE_CHECK
    global EXPERTS_NAME
    global WORKING_H
    global WORKING_W
    global RUN_TYPE

    if len(argv) < 8:
        status = 0
        status_code = 'Incorrect usage'
        return status, status_code

    RUN_TYPE = np.int32(argv[1])

    if not argv[2] == '-':
        INPUT_PATH = argv[2]
    if not argv[3] == '-':
        OUTPUT_PATH = argv[3]

    ENABLE_DOUBLE_CHECK = np.int32(argv[4])
    if not argv[5] == '0':
        WORKING_H = np.int32(argv[5])
    if not argv[6] == '0':
        WORKING_W = np.int32(argv[6])

    if not os.path.exists(INPUT_PATH) or len(os.listdir(INPUT_PATH)) == 0:
        status = 0
        status_code = 'Invalid input path'
        return status, status_code

    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    potential_experts = argv[7:]
    if argv[7] == 'all':
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
        return status, status_code

    for exp_name in EXPERTS_NAME:
        exp_out_path = os.path.join(OUTPUT_PATH, exp_name)
        if os.path.exists(exp_out_path):
            shutil.rmtree(exp_out_path)
        os.mkdir(exp_out_path)

    status = 1
    status_code = ''
    return status, status_code


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
    elif exp_name == 'edges_dexined':
        return experts.edges_expert.EdgesModel(full_expert=True)
    elif exp_name == 'normals_xtc':
        return experts.normals_expert.SurfaceNormalsModel(full_expert=True)
    elif exp_name == 'saliency_seg_egnet':
        return experts.saliency_seg_expert.SaliencySegmModel(full_expert=True)
    elif exp_name == 'rgb':
        return None


def get_image(img_path):
    img = cv2.imread(img_path)

    orig_h = img.shape[0]
    orig_w = img.shape[1]

    img = cv2.resize(img, (WORKING_W, WORKING_H), cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img, orig_h, orig_w


def get_pil_image(img_path):
    img = Image.open(img_path)

    orig_h, orig_w = img.size

    img = img.resize((WORKING_H, WORKING_W))

    return img, orig_h, orig_w


def process_and_save_samples():
    fileslist = os.listdir(INPUT_PATH)
    fileslist.sort()

    #fileslist = fileslist[0:10]

    for exp_name in EXPERTS_NAME:
        expert = get_expert(exp_name)
        exp_out_path = os.path.join(OUTPUT_PATH, exp_name)

        for filename in fileslist:
            img_path = os.path.join(INPUT_PATH, filename)

            if exp_name == 'depth_sgdepth' or exp_name == 'normals_xtc':
                img, _, _ = get_pil_image(img_path)
            else:
                img, _, _ = get_image(img_path)

            if expert == None:
                result = np.moveaxis(img, 2, 0)
            else:
                result = expert.apply_expert([img])[0]

            out_path = os.path.join(exp_out_path,
                                    filename.replace('.png', '.npy'))
            np.save(out_path, result)


if __name__ == "__main__":
    status, status_code = check_arguments_and_init_paths(sys.argv)
    if status == 0:
        sys.exit(status_code + '\n' + usage_str)

    if RUN_TYPE == 0:
        process_and_save_samples()