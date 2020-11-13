import glob
import os
import pathlib

import numpy as np
import torch
from PIL import Image
from torch import nn
from tqdm import tqdm
from utils.utils import img_for_plot

no_generated = 3000


def generate_experts_output(experts, config):
    '''
        ex.
            generate_experts_output([RGBModel(full_expert=True)], config)
    '''
    RGBS_PATH = config.get('Paths', 'RGBS_PATH')
    EXPERTS_OUTPUT_PATH = config.get('Paths', 'EXPERTS_OUTPUT_PATH')
    TRAIN_PATH = config.get('Paths', 'TRAIN_PATH')
    VALID_PATH = config.get('Paths', 'VALID_PATH')

    pattern = "%s/*/*00001.jpg"  # train/val:
    # pattern = "%s/*/*333.jpg"  # train/val:
    # pattern = "%s/*/*0050.jpg"  # train/val:
    # pattern = "%s/*/*0111.jpg"  # train/val:

    train_dir = "%s/%s" % (RGBS_PATH, TRAIN_PATH)
    valid_dir = "%s/%s" % (RGBS_PATH, VALID_PATH)
    for rgbs_dir_path in [train_dir, valid_dir]:
        count = 0
        rgb_paths = sorted(glob.glob(pattern % rgbs_dir_path))
        for rgb_path in tqdm(rgb_paths):
            img = Image.open(rgb_path)
            for expert in experts:
                fname = rgb_path.replace(
                    "/data/tracking-vot/",
                    "%s/%s/" % (EXPERTS_OUTPUT_PATH, expert.str_id)).replace(
                        ".jpg", ".npy")
                save_folder = os.path.dirname(fname)
                if not os.path.exists(save_folder):
                    pathlib.Path(save_folder).mkdir(parents=True,
                                                    exist_ok=True)
                if not os.path.exists(fname):
                    # e_out shape: expert.n_maps x 256 x 256
                    e_out = expert.apply_expert_one_frame(img)
                    np.save(fname, e_out)

                count += 1
            if count > no_generated * len(experts):
                break
        print("Done:", rgbs_dir_path, "pattern:", pattern, "no files",
              "rgb_paths:", len(rgb_paths))


def generate_experts_output_with_time(experts, config):
    '''
    ex.
            generate_experts_output_with_time([Tracking1Model(full_expert=True)], config)
    '''
    RGBS_PATH = config.get('Paths', 'RGBS_PATH')
    EXPERTS_OUTPUT_PATH = config.get('Paths', 'EXPERTS_OUTPUT_PATH')
    TRAIN_PATH = config.get('Paths', 'TRAIN_PATH')
    VALID_PATH = config.get('Paths', 'VALID_PATH')

    ends_with = "0001.jpg"  # train/val:
    # ends_with = "1333.jpg"  # train/val:
    # ends_with = "0333.jpg"  # train/val:
    # ends_with = "0050.jpg"  # train/val:
    # ends_with = "0111.jpg"  # train/val:
    train_dir = "%s/%s" % (RGBS_PATH, TRAIN_PATH)
    valid_dir = "%s/%s" % (RGBS_PATH, VALID_PATH)
    for rgbs_dir_path in [train_dir, valid_dir]:
        all_dirs = sorted(os.listdir(rgbs_dir_path))
        count = 0

        for crt_video_dir in tqdm(all_dirs):
            if len(
                    glob.glob("%s/%s/*%s" %
                              (rgbs_dir_path, crt_video_dir, ends_with))) == 0:
                continue

            # skip already generated outputs:
            already_gen = True
            for expert in experts:
                pattern = "%s/%s/*%s" % (rgbs_dir_path, crt_video_dir,
                                         ends_with)
                fpaths = glob.glob(pattern)
                for path in fpaths:
                    path = path.replace(
                        "/data/tracking-vot/",
                        "%s/%s/" % (EXPERTS_OUTPUT_PATH, expert.str_id))

                    npy_path = path.replace(".jpg", ".npy")
                    if not os.path.exists(npy_path):
                        already_gen = False
            if already_gen:
                count += 1
                continue

            all_imgs = sorted(
                glob.glob("%s/%s/*.jpg" % (rgbs_dir_path, crt_video_dir)))
            rgb_frames = []
            for rgb_path in all_imgs:
                img = Image.open(rgb_path)
                rgb_frames.append(img)

                # needs clear pattern, without *
                if rgb_path.endswith(ends_with):
                    break

            for expert in experts:
                fname = rgb_path.replace(
                    "/data/tracking-vot/",
                    "%s/%s/" % (EXPERTS_OUTPUT_PATH, expert.str_id)).replace(
                        ".jpg", ".npy")
                save_folder = os.path.dirname(fname)
                if not os.path.exists(save_folder):
                    pathlib.Path(save_folder).mkdir(parents=True,
                                                    exist_ok=True)
                if not os.path.exists(fname):
                    # TODO: doar de tracking?!! (to change)
                    with open(
                            "%s/%s/groundtruth.txt" %
                        (rgbs_dir_path, crt_video_dir), "r") as fd:
                        start_bbox_str = fd.readline()
                        start_bbox = [
                            float(x) for x in start_bbox_str.replace(
                                "\n", "").split(",")
                        ]

                    # e_out shape: expert.n_maps x 256 x 256
                    e_out = expert.apply_expert_for_last_map(
                        rgb_frames, start_bbox)

                    np.save(fname, e_out)
                count += 1

            if count > no_generated * len(experts):
                break
        print("Done:", rgbs_dir_path, "ends_with:", ends_with)
