# https://github.com/EPFL-VILAB/XTConsistency
import os

import cv2
import numpy as np
import PIL
import torch
from torchvision import transforms

from experts.xtc.unet import UNet

W, H = 256, 256

current_dir_name = os.path.dirname(os.path.realpath(__file__))
normals_model_path = os.path.join(current_dir_name, 'models/normals_xtc.pth')


class SurfaceNormalsXTC():
    def __init__(self, full_expert=True):
        if full_expert:
            model_path = normals_model_path  #"experts/models/rgb2normal_consistency.pth"
            self.model = UNet()
            model_state_dict = torch.load(model_path)
            self.model.load_state_dict(model_state_dict)
            self.model.eval()

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            self.model.to(device)

            self.trans_totensor = transforms.Compose([
                #transforms.Resize(W * 2, interpolation=PIL.Image.BILINEAR),
                #transforms.CenterCrop(W),
                transforms.ToTensor()
            ])
        self.domain_name = "normals"
        self.n_maps = 3
        self.str_id = "xtc"
        self.identifier = self.domain_name + "_" + self.str_id

    def apply_expert_batch(self, batch_rgb_frames):
        batch_rgb_frames = batch_rgb_frames.permute(0, 3, 1, 2) / 255.
        normals_maps = self.model(batch_rgb_frames.to(self.device))
        # normals_maps = normals_maps.clamp(
        #     min=0, max=1).data.cpu().float().numpy()
        norm = normals_maps.norm(dim=1)
        rez = normals_maps / norm[:, None]
        rez = rez.data.cpu().numpy().astype('float32')
        return rez
