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


for domain_idx, domain in enumerate(DOMAIN_EXPERTS):
    PATH_EXPERT = "/data/multi-domain-graph-2/datasets/datasets_preproc_exp/%s/%s/%s/" % (
        DATASET, SPLIT, domain)
    PATH_ENS1 = "/data/multi-domain-graph-2/datasets/datasets_preproc_ens_iter1/%s/%s/%s/" % (
        DATASET, SPLIT, domain)
    PATH_GT = "/data/multi-domain-graph-2/datasets/datasets_preproc_gt/%s/%s/%s/" % (
        DATASET, SPLIT, domain[:-4])
    count_all = 0
    count_better = 0

    for idx in range(0, 960):
        expert = np.load(os.path.join(PATH_EXPERT, "%08d.npy" % idx))
        ens1 = np.load(os.path.join(PATH_ENS1, "%08d.npy" % idx))
        gt = np.load(os.path.join(PATH_GT, "%08d.npy" % idx))

        dist_gt_exp = abs((gt - expert).sum(axis=0))
        dist_gt_ens1 = abs((gt - ens1).sum(axis=0))

        count_better += (dist_gt_exp - dist_gt_ens1 > -0.001).sum()
        count_all += dist_gt_exp.size
    print(domain, "Better: %.2f%%" % (count_better * 100 / count_all))
