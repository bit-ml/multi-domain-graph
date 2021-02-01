import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

device = torch.device('cpu')

SURFNORM_KERNEL = torch.from_numpy(
    np.array([
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    ]))[:, np.newaxis, ...].to(dtype=torch.float32, device=device)


def depth_to_surface_normals(depth, surfnorm_scalar=256):
    with torch.no_grad():
        surface_normals = F.conv2d(depth,
                                   surfnorm_scalar * SURFNORM_KERNEL,
                                   padding=1)
        surface_normals[:, 2, ...] = 1
        surface_normals = surface_normals / surface_normals.norm(dim=1,
                                                                 keepdim=True)
    return surface_normals


split_name = "test"
depth_path = "/data/multi-domain-graph-6/datasets/replica_raw/depth"
normals_path = "/data/multi-domain-graph-6/datasets/replica_raw/normals"

depth_full_path = os.path.join(depth_path, split_name)
normals_full_path = os.path.join(normals_path, split_name)
os.makedirs(normals_full_path, exist_ok=True)

all_files = os.listdir(depth_full_path)
all_files.sort()

for depth_file in all_files:
    depth_img_path = os.path.join(depth_full_path, depth_file)
    normals_img_path = os.path.join(normals_full_path, depth_file)

    depth_img = Image.open(depth_img_path)
    depth_img_th = torch.from_numpy(np.array(depth_img)) / 255.
    normals_img_th = depth_to_surface_normals(depth_img_th[None, None])

    # normals_img.save(normals_img_path)
    normals_img = normals_img_th.data.cpu().numpy()[0]

    normals_img[0] = np.round((0.5 * normals_img[0] + 0.5) * 255)
    normals_img[1] = np.round((0.5 * normals_img[1] + 0.5) * 255)
    normals_img[2] = np.round((0.5 * normals_img[2] + 0.5) * 255)
    normals_img = normals_img.astype(np.uint8).transpose(1, 2, 0)

    Image.fromarray(normals_img).save("abc.png")

    break
