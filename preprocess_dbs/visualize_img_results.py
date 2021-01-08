import os
import shutil
import sys
import cv2

import numpy as np

EXPERTS_PATH = r'/data/multi-domain-graph/datasets/datasets_preproc_exp'
GT_PATH = r'/data/multi-domain-graph/datasets/datasets_preproc_gt'

DATASET_PATH = r'taskonomy/sample-model'

OUTPUT_PATH = r'/root/tmp/visualize_dataset'

usage_str = 'python visualize_img_results.py dataset_path'

# generate colors for sseg maps
np.random.seed(1)
r = np.random.randint(0, 255, 100)
g = np.random.randint(0, 255, 100)
b = np.random.randint(0, 255, 100)
sseg_colors = np.concatenate((r[:, None], g[:, None], b[:, None]), 1)


def get_sseg_img(img):
    n_classes = min(100, img.shape[0])
    img_out = np.zeros((3, ) + img.shape[1:3])
    for i in range(n_classes):
        img_ = img[i, :, :]
        img_out[0, :, :] = img_ * sseg_colors[i, 0]
        img_out[1, :, :] = img_ * sseg_colors[i, 1]
        img_out[2, :, :] = img_ * sseg_colors[i, 2]
    return img_out


def img_for_plot(img):
    c, h, w = img.shape
    if c == 1:
        img = img.repeat(3, 0)
    if c == 2:
        img = np.concatenate(img, np.zeros((1, h, w)), 0)
    if c > 3:
        img = get_sseg_img(img)

    if np.max(img) > 1:
        min_val = np.min(img)
        max_val = np.max(img)
        img = (img - min_val) / (max_val - min_val)

    img = np.moveaxis(img, 0, -1)
    img = img * 255
    return img


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
    if len(sys.argv) != 2:
        sys.exit('incorrect usage [%s]' % usage_str)
    if not sys.argv[1] == '-':
        DATASET_PATH = sys.argv[1]

    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH)

    db_exp_path = os.path.join(EXPERTS_PATH, DATASET_PATH)
    db_gt_path = os.path.join(GT_PATH, DATASET_PATH)
    gt_rgbs_path = os.path.join(db_gt_path, 'rgb')

    gts = os.listdir(db_gt_path)
    experts = os.listdir(db_exp_path)

    #experts = ['normals_xtc']

    exp_results_path = []
    gt_results_path = []

    for exp_name in experts:
        gt_name = None
        for i in range(len(gts)):
            gt_name_ = gts[i]
            if exp_name[0:len(gt_name_)] == gt_name_:
                gt_name = gts[i]
        exp_results_path.append(os.path.join(db_exp_path, exp_name))
        if gt_name == None:
            gt_results_path.append(None)
        else:
            gt_results_path.append(os.path.join(db_gt_path, gt_name))

    print(experts)

    filenames = os.listdir(gt_rgbs_path)
    filenames.sort()
    filenames = filenames[0:50]
    for filename in filenames:
        img_path = os.path.join(gt_rgbs_path, filename)
        img = np.load(img_path)

        img = img_for_plot(img)
        img = np.concatenate((img, np.zeros((256, 256, 3))), 1)

        for domains in zip(exp_results_path, gt_results_path):
            exp_res_path, gt_res_path = domains

            exp_res = np.load(os.path.join(exp_res_path, filename))
            if gt_res_path == None:
                gt_res = np.zeros((3, 256, 256))
            else:
                gt_res = np.load(os.path.join(gt_res_path, filename))
            exp_res = img_for_plot(exp_res)
            gt_res = img_for_plot(gt_res)
            line = np.concatenate((exp_res, gt_res), 1)
            img = np.concatenate((img, line), 0)

        cv2.imwrite(
            os.path.join(OUTPUT_PATH, filename.replace('.npy', '.png')),
            np.uint8(img))
