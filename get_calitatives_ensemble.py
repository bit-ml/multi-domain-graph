import os

import numpy as np
from PIL import Image

DATASET = "replica"
SPLIT = "test"
PATTERN = "%05d"
# DOMAIN = "depth_n_1"
DOMAIN = "normals"
DOMAIN_EXP = "%s_xtc" % DOMAIN

PATH_RGB = "/data/multi-domain-graph-2/datasets/datasets_preproc_gt/%s/%s/rgb/" % (
    DATASET, SPLIT)
PATH_EXPERT = "/data/multi-domain-graph-2/datasets/datasets_preproc_exp/%s/%s/%s/" % (
    DATASET, SPLIT, DOMAIN_EXP)
PATH_ENS1 = "/data/multi-domain-graph-2/datasets/datasets_preproc_ens_iter1/%s/%s/%s/" % (
    DATASET, SPLIT, DOMAIN_EXP)
PATH_GT = "/data/multi-domain-graph-2/datasets/datasets_preproc_gt/%s/%s/%s/" % (
    DATASET, SPLIT, DOMAIN)

os.system("mkdir -p calitatives/ensemble/%s/" % DOMAIN)
for idx in [1]:
    rgb = np.load(os.path.join(PATH_RGB, ("%s.npy" % PATTERN) % idx))
    expert = np.load(os.path.join(PATH_EXPERT, "%08d.npy" % idx))
    ens1 = np.load(os.path.join(PATH_ENS1, "%08d.npy" % idx))
    gt = np.load(os.path.join(PATH_GT, "%08d.npy" % idx))

    rgb_pil = Image.fromarray((rgb.transpose(1, 2, 0) * 255.).astype(np.uint8))
    rgb_pil.save("calitatives/ensemble/%s/rgb_%i.png" % (DOMAIN, idx))

    expert_pil = (expert.transpose(1, 2, 0) * 255.).astype(np.uint8)
    if expert_pil.shape[-1] == 1:
        expert_pil = expert_pil[..., 0]
    Image.fromarray(expert_pil).save("calitatives/ensemble/%s/expert_%i.png" %
                                     (DOMAIN, idx))

    ens1_pil = (ens1.transpose(1, 2, 0) * 255.).astype(np.uint8)
    if ens1_pil.shape[-1] == 1:
        ens1_pil = ens1_pil[..., 0]
    Image.fromarray(ens1_pil).save("calitatives/ensemble/%s/ens1_%i.png" %
                                   (DOMAIN, idx))

    gt_pil = (gt.transpose(1, 2, 0) * 255.).astype(np.uint8)
    if gt_pil.shape[-1] == 1:
        gt_pil = gt_pil[..., 0]
    Image.fromarray(gt_pil).save("calitatives/ensemble/%s/gt_%i.png" %
                                 (DOMAIN, idx))
