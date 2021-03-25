import os

import numpy as np
import torch
from PIL import Image
from skimage import color

DATASET = "replica"
SPLIT = "test"
PATTERN = "%05d"
DOMAIN_EXPERTS = [
    "superpixel_fcn", "sem_seg_hrnet", "normals_xtc", "depth_n_1_xtc",
    "cartoon_wb", "edges_dexined"
]

DOMAIN_MIN_SCORE = [0, 0, 0.02, 0.03, 0., 0.]
DOMAIN_MAX_SCORE = [100, 0.3, 100, 100, 100., 100.]

PATH_RGB = "/data/multi-domain-graph-2/datasets/datasets_preproc_gt/%s/%s/rgb/" % (
    DATASET, SPLIT)

COLORS_SHORT = ('red', 'blue', 'yellow', 'magenta', 'green', 'indigo',
                'darkorange', 'cyan', 'pink', 'yellowgreen', 'chocolate',
                'lightsalmon', 'lime', 'silver', 'gainsboro', 'gold', 'coral',
                'aquamarine', 'lightcyan', 'oldlace', 'darkred', 'snow')


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


save_folder = "calitatives/expvscshift/"
os.system("mkdir -p %s" % save_folder)
os.system("rm -rf %s/*.png" % save_folder)

for idx in range(685, 686):
    skip_sample = False
    for domain_idx, domain in enumerate(DOMAIN_EXPERTS):
        PATH_EXPERT = "/data/multi-domain-graph-2/datasets/datasets_preproc_exp/%s/%s/%s/" % (
            DATASET, SPLIT, domain)
        PATH_ENS1 = "/data/multi-domain-graph-2/datasets/datasets_preproc_ens_iter1/%s/%s/%s/" % (
            DATASET, SPLIT, domain)

        expert = np.load(os.path.join(PATH_EXPERT, "%08d.npy" % idx))
        ens1 = np.load(os.path.join(PATH_ENS1, "%08d.npy" % idx))

        if domain in ["depth_n_1_xtc", "normals_xtc"]:
            PATH_GT = "/data/multi-domain-graph-2/datasets/datasets_preproc_gt/%s/%s/%s/" % (
                DATASET, SPLIT, domain[:-4])
            gt = np.load(os.path.join(PATH_GT, "%08d.npy" % idx))
            dist = dist_masked_gt(gt, expert, ens1)
        else:
            dist = abs(expert - ens1).mean()

        if dist < DOMAIN_MIN_SCORE[domain_idx] or dist > DOMAIN_MAX_SCORE[
                domain_idx]:
            skip_sample = True
            print(idx, "slab", domain, dist)
            break

    if skip_sample:
        continue

    print(idx)
    sz = 4
    result_img = np.zeros((2 * 256 + 3 * sz, 6 * 256 + sz * 7, 3),
                          dtype=np.uint8)
    for domain_idx, domain in enumerate(DOMAIN_EXPERTS):
        PATH_EXPERT = "/data/multi-domain-graph-2/datasets/datasets_preproc_exp/%s/%s/%s/" % (
            DATASET, SPLIT, domain)
        PATH_ENS1 = "/data/multi-domain-graph-2/datasets/datasets_preproc_ens_iter1/%s/%s/%s/" % (
            DATASET, SPLIT, domain)
        expert = np.load(os.path.join(PATH_EXPERT, "%08d.npy" % idx))
        ens1 = np.load(os.path.join(PATH_ENS1, "%08d.npy" % idx))
        dist = abs(expert - ens1).mean()
        print("\t%10s - %.3f" % (domain, dist))
        if domain == "sem_seg_hrnet":
            expert = sseg_map_to_img(expert)
            ens1 = sseg_map_to_img(ens1)

        expert_pil = (expert.transpose(1, 2, 0) * 255.).astype(np.uint8)
        if expert_pil.shape[-1] == 1:
            expert_pil = expert_pil.repeat(repeats=3, axis=2)
        result_img[sz:(256 + sz),
                   sz + (256 + sz) * domain_idx:(256 + sz) * domain_idx + 256 +
                   sz] = expert_pil

        ens1_pil = (ens1.transpose(1, 2, 0) * 255.).astype(np.uint8)
        if ens1_pil.shape[-1] == 1:
            ens1_pil = ens1_pil.repeat(repeats=3, axis=2)
        result_img[(256 + 2 * sz):(256 + 2 * sz) + 256,
                   (256 + sz) * domain_idx + sz:(256 + sz) * domain_idx + 256 +
                   sz] = ens1_pil

    Image.fromarray(result_img).save("%s/%d_compare.png" % (save_folder, idx))
    # rgb = np.load(os.path.join(PATH_RGB, ("%s.npy" % PATTERN) % idx))
    # rgb_pil = Image.fromarray((rgb.transpose(1, 2, 0) * 255.).astype(np.uint8))
    # rgb_pil.save("%s/%d_rgb.png" % (save_folder, idx))
