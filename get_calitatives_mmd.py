import os

import numpy as np
from PIL import Image

# DATASET = "taskonomy"
# SPLIT = "tiny-test"
# PATTERN = "%08d"

DATASET = "replica"
SPLIT = "test"
PATTERN = "%05d"

# DATASET = "hypersim"
# SPLIT = "test"
# PATTERN = "%08d"

PATH_RGB = "/data/multi-domain-graph-2/datasets/datasets_preproc_gt/%s/%s/rgb/" % (
    DATASET, SPLIT)
PATH_NORMALS = "/data/multi-domain-graph-2/datasets/datasets_preproc_gt/%s/%s/normals/" % (
    DATASET, SPLIT)
PATH_DEPTH = "/data/multi-domain-graph-2/datasets/datasets_preproc_gt/%s/%s/depth_n_1/" % (
    DATASET, SPLIT)

for idx in [100]:
    rgb = np.load(os.path.join(PATH_RGB, ("%s.npy" % PATTERN) % idx))
    normals = np.load(os.path.join(PATH_NORMALS, "%08d.npy" % idx))
    depth = np.load(os.path.join(PATH_DEPTH, "%08d.npy" % idx))

    rgb_pil = Image.fromarray((rgb.transpose(1, 2, 0) * 255.).astype(np.uint8))
    rgb_pil.save("calitatives/mmd/%s_rgb_%i.png" % (DATASET, idx))

    normals_pil = Image.fromarray(
        (normals.transpose(1, 2, 0) * 255.).astype(np.uint8))
    normals_pil.save("calitatives/mmd/%s_normals_%i.png" % (DATASET, idx))

    depth_pil = Image.fromarray((depth[0] * 255.).astype(np.uint8))
    depth_pil.save("calitatives/mmd/%s_depth_%i.png" % (DATASET, idx))
