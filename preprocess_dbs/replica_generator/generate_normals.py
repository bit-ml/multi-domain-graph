import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

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


def generate_normals():
    split_name = "test"
    depth_full_path = "/data/multi-domain-graph-6/datasets/replica_raw/%s/depth" % split_name
    normals_full_path = "/data/multi-domain-graph-6/datasets/replica_raw/%s/normals" % split_name

    os.makedirs(normals_full_path, exist_ok=True)

    all_files = os.listdir(depth_full_path)
    all_files.sort()

    for depth_file in tqdm(all_files):
        # print("id", depth_file)
        file_id = int(depth_file.replace(".npy", ""))
        depth_img_path = os.path.join(depth_full_path, depth_file)

        # save normals
        depth_img = 1 - np.load(depth_img_path) / 14.
        depth_img_th = torch.from_numpy(depth_img)

        normals_img_th = depth_to_surface_normals(depth_img_th[None, None])[0]

        normals_img_th[0] = (0.5 * normals_img_th[0] + 0.5)
        normals_img_th[1] = (0.5 * normals_img_th[1] + 0.5)
        normals_img_th[2] = (0.5 * normals_img_th[2] + 0.5)

        permute = [2, 0, 1]
        normals_img = normals_img_th[permute].data.cpu().numpy()

        # # 1. SAVE Normals npy
        # normals_img_path = os.path.join(normals_full_path, depth_file)
        # np.save(normals_img_path, normals_img)

        # # 2. SAVE RGB npy
        # images_full_path = "/data/multi-domain-graph-6/datasets/replica_raw/%s/images" % split_name
        # rgb_full_path = "/data/multi-domain-graph-6/datasets/replica_raw/%s/rgb" % split_name
        # images_img_path = os.path.join(images_full_path, "%05d.png" % file_id)
        # rgb_img_path = os.path.join(rgb_full_path, "%05d.npy" % file_id)
        # rgb_img = np.array(Image.open(images_img_path),
        #                    dtype=np.float32).transpose(2, 0, 1)[:3] / 255.
        # np.save(rgb_img_path, rgb_img)

        # # 3. Check PNGs: rgb vs normals
        # Image.fromarray((depth_img * 255).astype(np.uint8)).save("depth.png")
        # Image.fromarray((normals_img * 255).astype(np.uint8).transpose(
        #     1, 2, 0)).save("normals_012.png")

        # Image.fromarray((normals_img[[2, 1, 0]] * 255).astype(
        #     np.uint8).transpose(1, 2, 0)).save("normals_210.png")
        # Image.fromarray((normals_img[[2, 0, 1]] * 255).astype(
        #     np.uint8).transpose(1, 2, 0)).save("normals_201.png")
        # Image.fromarray((normals_img[[1, 2, 0]] * 255).astype(
        #     np.uint8).transpose(1, 2, 0)).save("normals_120.png")
        # Image.fromarray((normals_img[[1, 0, 2]] * 255).astype(
        #     np.uint8).transpose(1, 2, 0)).save("normals_102.png")

        # Image.fromarray((normals_img[[0, 2, 1]] * 255).astype(
        #     np.uint8).transpose(1, 2, 0)).save("normals_021.png")
        # print("inca una", depth_file)

        break


def check_alingment():
    split_name = "test"
    depth_gt_full_path = "/data/multi-domain-graph-6/datasets/datasets_preproc_gt/replica/%s/depth" % split_name
    normals_gt_full_path = "/data/multi-domain-graph-6/datasets/datasets_preproc_gt/replica/%s/normals" % split_name

    normals_exp_full_path = "/data/multi-domain-graph-6/datasets/datasets_preproc_exp/replica/%s/normals_xtc" % split_name
    depth_exp_full_path = "/data/multi-domain-graph-6/datasets/datasets_preproc_exp/replica/%s/depth_xtc" % split_name

    images_full_path = "/data/multi-domain-graph-6/datasets/replica_raw/%s/images" % split_name
    rgb_full_path = "/data/multi-domain-graph-6/datasets/replica_raw/%s/rgb" % split_name

    for i in range(10):
        depth_img_path = os.path.join(depth_gt_full_path, "%05d.npy" % i)
        depth_exp_img_path = os.path.join(depth_exp_full_path, "%08d.npy" % i)

        normals_img_path = os.path.join(normals_exp_full_path, "%08d.npy" % i)
        normals_gt_img_path = os.path.join(normals_gt_full_path,
                                           "%05d.npy" % i)

        images_img_path = os.path.join(images_full_path, "%05d.png" % i)
        rgb_img_path = os.path.join(rgb_full_path, "%05d.npy" % i)

        # verify
        depth_img = np.load(depth_img_path)
        depth_exp_img = np.load(depth_exp_img_path)

        normals_img = np.load(normals_img_path)
        normals_gt_img = np.load(normals_gt_img_path)

        Image.fromarray(
            (depth_img[0] * 255).astype(np.uint8)).save("depth_gt.png")
        Image.fromarray(
            (depth_exp_img[0] * 255).astype(np.uint8)).save("depth_xtc.png")

        Image.fromarray((normals_img * 255).astype(np.uint8).transpose(
            1, 2, 0)).save("normals_xtc.png")
        Image.fromarray((normals_gt_img * 255).astype(np.uint8).transpose(
            1, 2, 0)).save("normals_gt.png")

        rgb_img = np.array(Image.open(images_img_path),
                           dtype=np.float32).transpose(2, 0, 1)[:3] / 255.
        # np.save(rgb_img_path, rgb_img)

        print("inca una")
        # break


# normals_npy_file = "/data/multi-domain-graph-6/datasets/datasets_preproc_gt/replica/test/normals/00000.npy"
# Image.fromarray((np.load(normals_npy_file) * 255.).astype(np.uint8).transpose(1, 2, 0)).save("normals.png")

# check_alingment()
generate_normals()
