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

import utils.visualize_of
import utils.visualize_vmos

# usage
# python main_save_experts path1 path2 enable_double_check exp1 exp2
#    path1                  [a path / '-'] - experts main path
#                                           - if '-' is provided, we use INPUT_PATH's default value
#    path2                  [a path / '-'] - visualization path
#                                           - if '-' is provided, we use OUTPUT_PATH's default value
#    path3                  [a path / '-'] - videos orig path
#                                           - if '-' is provided, we use VIDEOS_ORIG_PATH's default value
#    db_name                dataset name; if '' it will be ignored
#                                       - if '-' is provided, we use DB_NAME's default value
#    subset_name            subset name; if '' it will be ignored
#                                       - if '-' is provided, we use SUBSET_NAME's default value
#    working_h              [nr>0 / 0] - desired image height
#    working_w              [nr>0 / 0] - desired image width

INPUT_PATH = r'/data/experts-output/'
OUTPUT_PATH = r'/root/experts_vis'
VIDEOS_ORIG_PATH = r'/data/tracking-vot'
WORKING_H = 256
WORKING_W = 256
DB_NAME = 'GOT-10k'
SUBSET_NAME = 'val'


def check_arguments_and_init_paths(argv):
    global INPUT_PATH
    global OUTPUT_PATH
    global VIDEOS_ORIG_PATH
    global WORKING_H
    global WORKING_W
    global DB_NAME
    global SUBSET_NAME

    if not argv[1] == '-':
        INPUT_PATH = argv[1]
    if not argv[2] == '-':
        OUTPUT_PATH = argv[2]
    if not argv[3] == '-':
        VIDEOS_ORIG_PATH = argv[3]
    if not argv[4] == '-':
        DB_NAME = argv[4]
    if not argv[5] == '-':
        SUBSET_NAME = argv[5]

    if not argv[6] == '0':
        WORKING_H = np.int32(argv[6])
    if not argv[7] == '0':
        WORKING_W = np.int32(argv[7])

    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    os.mkdir(OUTPUT_PATH)

    if not os.path.exists(INPUT_PATH):
        status = 0
        status_code = 'Invalid input path: %s' % INPUT_PATH
        return status, status_code

    if len(os.listdir(INPUT_PATH)) == 0:
        status = 0
        status_code = 'Empty input path: %s' % INPUT_PATH
        return status, status_code

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


def build_display_img(frame, exp_results):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    max_nr_channels = 0
    for exp_res in exp_results:
        max_nr_channels = max(max_nr_channels, exp_res.shape[0])
    frame = np.concatenate(
        (frame,
         np.zeros(
             ((max_nr_channels - 1) * frame.shape[0], frame.shape[1]))), 0)
    for exp_res in exp_results:
        img = np.reshape(
            exp_res, (exp_res.shape[0] * exp_res.shape[1], exp_res.shape[2]))
        min_val = np.min(img)
        max_val = np.max(img)
        d = max_val - min_val
        if d < 0.00000001:
            d = 1
        img = (img - min_val) / d
        img = img * 255
        if max_nr_channels > exp_res.shape[0]:
            img = np.concatenate(
                (img,
                 np.zeros(
                     ((max_nr_channels - exp_res.shape[0]) * exp_res.shape[1],
                      exp_res.shape[2]))), 0)
        frame = np.concatenate((frame, img), 1)
    return frame


def visualize_experts():
    #import pdb
    #pdb.set_trace()
    experts_name = os.listdir(INPUT_PATH)
    experts_name.sort()
    videos_name = os.listdir(
        os.path.join(INPUT_PATH, experts_name[0], DB_NAME, SUBSET_NAME))
    videos_name.sort()

    for video_name in videos_name:
        frames, out_filenames = get_rgb_video_frames(
            os.path.join(VIDEOS_ORIG_PATH, DB_NAME, SUBSET_NAME, video_name))
        vid_out_path = os.path.join(OUTPUT_PATH, video_name)
        os.mkdir(vid_out_path)

        for frame_idx in range(len(frames)):
            exp_results = []
            for exp_name in experts_name:
                exp_res_path = os.path.join(INPUT_PATH, exp_name)
                if not DB_NAME == '':
                    exp_res_path = os.path.join(exp_res_path, DB_NAME)
                if not SUBSET_NAME == '':
                    exp_res_path = os.path.join(exp_res_path, SUBSET_NAME)

                exp_res_path = os.path.join(exp_res_path, video_name,
                                            '%08d.npy' % frame_idx)
                #import pdb
                #pdb.set_trace()
                if os.path.exists(exp_res_path):
                    exp_results.append(np.load(exp_res_path))
                else:
                    exp_results.append(
                        np.zeros((1, frames[frame_idx].shape[0],
                                  frames[frame_idx].shape[1])))

            #import pdb
            #pdb.set_trace()
            img = build_display_img(frames[frame_idx], exp_results)

            cv2.imwrite(os.path.join(vid_out_path, '%08d.png' % frame_idx),
                        np.uint8(img))


if __name__ == "__main__":
    status, status_code = check_arguments_and_init_paths(sys.argv)
    if status == 0:
        sys.exit(status_code)

    visualize_experts()
