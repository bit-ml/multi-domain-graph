import os

import numpy as np
import torch
from PIL import Image
from skimage import color

DATASET = "replica"
SPLIT = "test"
PATTERN = "%05d"

DOMAIN_EXPERTS_GT = ["normals_xtc", "depth_n_1_xtc"]
# DOMAIN_EXPERTS = [
#     "normals_xtc", "depth_n_1_xtc", "superpixel_fcn", "sem_seg_hrnet",
#     "cartoon_wb", "edges_dexined"
# ]
DOMAIN_EXPERTS = [
    "superpixel_fcn", "sem_seg_hrnet", "cartoon_wb", "edges_dexined"
]

DOMAIN_MIN_SCORE = [0.01, 0.01, 0.01, 0.01]
DOMAIN_MAX_SCORE = [100, 100, 100, 100]

PATH_RGB = "/data/multi-domain-graph-2/datasets/datasets_preproc_gt/%s/%s/rgb/" % (
    DATASET, SPLIT)

COLORS_SHORT = ('red', 'blue', 'yellow', 'magenta', 'green', 'indigo',
                'darkorange', 'cyan', 'pink', 'yellowgreen', 'chocolate',
                'lightsalmon', 'lime', 'silver', 'gainsboro', 'gold', 'coral',
                'aquamarine', 'lightcyan', 'oldlace', 'darkred', 'snow')


def to_pil(image_tensor):
    pil_ = (image_tensor.transpose(1, 2, 0) * 255.).astype(np.uint8)
    if pil_.shape[-1] == 1:
        pil_ = pil_.repeat(repeats=3, axis=2)
    return pil_


def sseg_map_to_img(img):
    all_classes = 12
    for idx in range(all_classes):
        img[0, 0, idx] = idx
        img[0, idx, 0] = idx

    result = color.label2rgb(img[0], colors=COLORS_SHORT,
                             bg_label=0).transpose(2, 0, 1)
    img = result.astype(np.float32)
    return img


def dist_masked_gt(gt, exp, pred):
    is_nan = gt != gt

    if is_nan.sum() > 0:
        l_target = gt.copy()

        bm = ~is_nan

        l_target[is_nan] = 0
        l_target = l_target * bm
        l_ens = pred * bm
        l_exp = exp * bm
    else:
        l_ens = pred
        l_exp = exp
        l_target = gt

    dist_exp = abs(l_target - l_exp).mean()
    dist_ens = abs(l_target - l_ens).mean()
    dist = dist_exp - dist_ens
    return dist


for domain in DOMAIN_EXPERTS:
    save_folder = "calitatives/video/%s/" % domain
    os.system("mkdir -p %s" % save_folder)
    os.system("rm -rf %s/*.png" % save_folder)


def run_for(frames_indices):
    for idx in frames_indices:
        for domain_idx, domain in enumerate(DOMAIN_EXPERTS):
            PATH_EXPERT = "/data/multi-domain-graph-2/datasets/datasets_preproc_exp/%s/%s/%s/" % (
                DATASET, SPLIT, domain)
            PATH_ENS1 = "/data/multi-domain-graph-2/datasets/datasets_preproc_ens_iter1/%s/%s/%s/" % (
                DATASET, SPLIT, domain)

            expert = np.load(os.path.join(PATH_EXPERT, "%08d.npy" % idx))
            ens1 = np.load(os.path.join(PATH_ENS1, "%08d.npy" % idx))

            if domain in DOMAIN_EXPERTS_GT:
                PATH_GT = "/data/multi-domain-graph-2/datasets/datasets_preproc_gt/%s/%s/%s/" % (
                    DATASET, SPLIT, domain[:-4])
                gt = np.load(os.path.join(PATH_GT, "%08d.npy" % idx))
                dist = dist_masked_gt(gt, expert, ens1)
            else:
                dist = abs(expert - ens1).mean()

            if DOMAIN_MAX_SCORE[domain_idx] > dist > DOMAIN_MIN_SCORE[
                    domain_idx]:
                sz = 4
                SIZE = 256
                sx = sz
                ex = sx + SIZE

                result_img = np.zeros((ex, 6 * SIZE + sz * 7, 3),
                                      dtype=np.uint8)

                print("\t%10s - %.3f" % (domain, dist))
                if domain == "sem_seg_hrnet":
                    expert = sseg_map_to_img(expert)
                    ens1 = sseg_map_to_img(ens1)

                if domain in DOMAIN_EXPERTS_GT:
                    num_pics = 5
                else:
                    num_pics = 3
                result_img = np.zeros(
                    (SIZE + 2 * sz, num_pics * SIZE + sz * (num_pics + 1), 3),
                    dtype=np.uint8)

                # RGB
                rgb = np.load(
                    os.path.join(PATH_RGB, ("%s.npy" % PATTERN) % idx))
                rgb_pil = (rgb.transpose(1, 2, 0) * 255.).astype(np.uint8)
                result_img[sx:ex, sz:(SIZE + sz)] = rgb_pil

                # Expert
                expert_pil = to_pil(expert)
                result_img[sx:ex,
                           sz + (SIZE + sz):(SIZE + sz) * 2] = expert_pil

                # CShift
                ens1_pil = to_pil(ens1)
                result_img[sx:ex,
                           sz + (SIZE + sz) * 2:(SIZE + sz) * 3] = ens1_pil

                if domain in DOMAIN_EXPERTS_GT:
                    # GT
                    gt_pil = to_pil(gt)
                    result_img[sx:ex,
                               sz + (SIZE + sz) * 3:(SIZE + sz) * 4] = gt_pil

                    # diff
                    dist_gt_exp = abs((gt - expert).sum(axis=0))
                    dist_gt_ens1 = abs((gt - ens1).sum(axis=0))

                    diff_pil = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
                    mask_better = dist_gt_exp > dist_gt_ens1
                    mask_worse = dist_gt_exp < dist_gt_ens1
                    diff_pil[mask_worse, 0] = 255
                    diff_pil[mask_better, 1] = 255
                    result_img[sx:ex,
                               sz + (SIZE + sz) * 4:(SIZE + sz) * 5] = diff_pil

                save_folder = "calitatives/video/%s/" % domain
                np.save("%s/%d" % (save_folder, idx), result_img)
                Image.fromarray(result_img).save("%s/%s_%d.png" %
                                                 (save_folder, domain, idx))


if __name__ == '__main__':
    # # global
    # domain_experts = [
    #     "normals_xtc", "depth_n_1_xtc", "superpixel_fcn", "sem_seg_hrnet",
    #     "cartoon_wb", "edges_dexined"
    # ]
    # min_score = [0.03, 0.06, 0.05, 0.02, 0.05, 0.03]
    # max_score = [100, 100, 100, 1., 100, 100]

    run_for(range(0, 960))
