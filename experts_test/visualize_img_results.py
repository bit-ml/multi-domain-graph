import os
import shutil
import sys
import cv2

import numpy as np

# usage : python visualize_img_results.py images_path experts_main_path output_path
# python visualize_img_results.py - - -

IMAGES_PATH = r'/data/multi-domain-graph/datasets/taskonomy/taskonomy-sample-model-1-master/rgb'
EXPERTS_MAIN_PATH = r'/data/multi-domain-graph/datasets/taskonomy/taskonomy-sample-model-1-master-experts'
OUTPUT_PATH = r'/root/tmp/visualize_experts_test'
WORKING_H = 256
WORKING_W = 256


def get_image(img_path):
    img = cv2.imread(img_path)

    orig_h = img.shape[0]
    orig_w = img.shape[1]

    img = cv2.resize(img, (WORKING_W, WORKING_H), cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img, orig_h, orig_w


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


if __name__ == "__main__":
    #global IMAGES_PATH
    #global EXPERTS_MAIN_PATH
    #global OUTPUT_PATH

    if not sys.argv[1] == '-':
        IMAGES_PATH = sys.argv[1]
    if not sys.argv[2] == '-':
        EXPERTS_MAIN_PATH = sys.argv[2]
    if not sys.argv[3] == '-':
        OUTPUT_PATH = sys.argv[3]

    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH)

    fileslist = os.listdir(IMAGES_PATH)
    fileslist.sort()

    experts = os.listdir(EXPERTS_MAIN_PATH)
    experts.sort()

    for filename in fileslist:
        img_path = os.path.join(IMAGES_PATH, filename)
        img, _, _ = get_image(img_path)
        exp_results = []
        for exp_name in experts:
            exp_res_path = os.path.join(EXPERTS_MAIN_PATH, exp_name,
                                        filename.replace('.png', '.npy'))
            if os.path.exists(exp_res_path):
                exp_results.append(np.load(exp_res_path))
            else:
                exp_results.append(np.zeros((1, WORKING_H, WORKING_W)))

        img = build_display_img(img, exp_results)

        cv2.imwrite(os.path.join(OUTPUT_PATH, filename), np.uint8(img))
