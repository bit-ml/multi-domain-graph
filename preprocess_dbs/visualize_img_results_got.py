import os
import shutil
import sys
import cv2

import numpy as np

sys.path.insert(0,
                os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import utils.visualize_of

MAIN_PATH = r'/data/multi-domain-graph/datasets/datasets_preproc_exp'

DATASET_PATH = r'GOT-10k/val'

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


def img_for_plot(img, exp_name):
    c, h, w = img.shape
    if c == 1:
        img = img.repeat(3, 0)
    if c == 2:
        if exp_name[0:2] == 'of':  # is flow
            img = np.moveaxis(img, 0, -1)
            img = utils.visualize_of.flow_to_image(img)
            img = np.moveaxis(img, 2, 0)
        else:
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

    db_path = os.path.join(MAIN_PATH, DATASET_PATH)
    experts = os.listdir(db_path)
    experts.sort()

    #experts = ['xtc_surface_normals']
    print(experts)

    idx = 0
    videos = os.listdir(os.path.join(db_path, experts[0]))
    videos.sort()
    videos = videos[0:50]
    for video_name in videos:
        img_path = os.path.join(db_path, 'rgb', video_name, '%08d.npy' % 1)
        img = np.load(img_path)
        img = img_for_plot(img, 'rgb')

        for exp_name in experts:
            exp_path = os.path.join(db_path, exp_name, video_name,
                                    '%08d.npy' % 1)
            exp_res = np.load(exp_path)
            exp_res = img_for_plot(exp_res, exp_name)

            img = np.concatenate((img, exp_res), 0)

        cv2.imwrite(os.path.join(OUTPUT_PATH, '%08d.png' % idx), np.uint8(img))
        idx = idx + 1