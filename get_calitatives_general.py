import os

import numpy as np
from PIL import Image

DATASET = "replica"
SPLIT = "test"
PATTERN = "%05d"

DOMAIN_EXPERTS = ["depth_n_1_xtc", "normals_xtc"]
DOMAIN_MIN_SCORE = [0.04, 0.05]

# DOMAIN_EXPERTS = ["normals_xtc"]
# DOMAIN_MIN_SCORE = [0.05]


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


save_folder = "calitatives/general/"
os.system("mkdir -p %s" % save_folder)
os.system("rm -rf %s/*.png" % save_folder)

for idx in [469, 86]:
    sz = 4
    SIZE = 256

    PATH_RGB = "/data/multi-domain-graph-2/datasets/datasets_preproc_gt/%s/%s/rgb/" % (
        DATASET, SPLIT)
    skip_sample = False

    for domain_idx, domain in enumerate(DOMAIN_EXPERTS):
        PATH_EXPERT = "/data/multi-domain-graph-2/datasets/datasets_preproc_exp/%s/%s/%s/" % (
            DATASET, SPLIT, domain)
        PATH_ENS1 = "/data/multi-domain-graph-2/datasets/datasets_preproc_ens_iter1/%s/%s/%s/" % (
            DATASET, SPLIT, domain)
        PATH_GT = "/data/multi-domain-graph-2/datasets/datasets_preproc_gt/%s/%s/%s/" % (
            DATASET, SPLIT, domain[:-4])

        expert = np.load(os.path.join(PATH_EXPERT, "%08d.npy" % idx))
        ens1 = np.load(os.path.join(PATH_ENS1, "%08d.npy" % idx))
        gt = np.load(os.path.join(PATH_GT, "%08d.npy" % idx))
        dist = dist_masked_gt(gt, expert, ens1)

        if dist < DOMAIN_MIN_SCORE[domain_idx]:
            # print(idx, domain, "dist", dist)
            skip_sample = True
            break

    if skip_sample:
        continue

    print(idx, "Good sample")
    # write to final image
    # rgb
    rgb = np.load(os.path.join(PATH_RGB, ("%s.npy" % PATTERN) % idx))
    rgb_pil = (rgb.transpose(1, 2, 0) * 255.).astype(np.uint8)
    result_img = np.zeros((2 * SIZE + 3 * sz, 5 * SIZE + sz * 6, 3),
                          dtype=np.uint8)
    result_img[sz:(SIZE + sz), sz:(SIZE + sz)] = rgb_pil
    result_img[sz + (SIZE + sz) * 1:(SIZE + sz) * 2, sz:(SIZE + sz)] = rgb_pil

    for domain_idx, domain in enumerate(DOMAIN_EXPERTS):
        PATH_EXPERT = "/data/multi-domain-graph-2/datasets/datasets_preproc_exp/%s/%s/%s/" % (
            DATASET, SPLIT, domain)
        PATH_ENS1 = "/data/multi-domain-graph-2/datasets/datasets_preproc_ens_iter1/%s/%s/%s/" % (
            DATASET, SPLIT, domain)
        PATH_GT = "/data/multi-domain-graph-2/datasets/datasets_preproc_gt/%s/%s/%s/" % (
            DATASET, SPLIT, domain[:-4])

        expert = np.load(os.path.join(PATH_EXPERT, "%08d.npy" % idx))
        ens1 = np.load(os.path.join(PATH_ENS1, "%08d.npy" % idx))
        gt = np.load(os.path.join(PATH_GT, "%08d.npy" % idx))

        # expert
        expert_pil = (expert.transpose(1, 2, 0) * 255.).astype(np.uint8)
        if expert_pil.shape[-1] == 1:
            expert_pil = expert_pil.repeat(repeats=3, axis=2)
        result_img[sz + (SIZE + sz) * domain_idx:(SIZE + sz) *
                   (domain_idx + 1),
                   sz + (SIZE + sz):(SIZE + sz) * 2] = expert_pil

        # ens
        ens1_pil = (ens1.transpose(1, 2, 0) * 255.).astype(np.uint8)
        if ens1_pil.shape[-1] == 1:
            ens1_pil = ens1_pil.repeat(repeats=3, axis=2)
        result_img[sz + (SIZE + sz) * domain_idx:(SIZE + sz) *
                   (domain_idx + 1),
                   sz + (SIZE + sz) * 2:(SIZE + sz) * 3] = ens1_pil
        # GT
        gt_pil = (gt.transpose(1, 2, 0) * 255.).astype(np.uint8)
        if gt_pil.shape[-1] == 1:
            gt_pil = gt_pil.repeat(repeats=3, axis=2)
        result_img[sz + (SIZE + sz) * domain_idx:(SIZE + sz) *
                   (domain_idx + 1),
                   sz + (SIZE + sz) * 3:(SIZE + sz) * 4] = gt_pil

        # diff
        dist_gt_exp = abs((gt - expert).sum(axis=0))
        dist_gt_ens1 = abs((gt - ens1).sum(axis=0))

        diff_pil = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        mask_better = dist_gt_exp > dist_gt_ens1
        mask_worse = dist_gt_exp < dist_gt_ens1
        diff_pil[mask_worse, 0] = 255
        diff_pil[mask_better, 1] = 255
        result_img[sz + (SIZE + sz) * domain_idx:(SIZE + sz) *
                   (domain_idx + 1),
                   sz + (SIZE + sz) * 4:(SIZE + sz) * 5] = diff_pil
    Image.fromarray(result_img).save("%s/%i_composed.png" % (save_folder, idx))
