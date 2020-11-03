# https://github.com/EPFL-VILAB/XTConsistency

import cv2
import numpy as np
import PIL
import torch
from torchvision import transforms

from experts.normals.unet import UNet

W, H = 256, 256


class SurfaceNormalsModel():
    def __init__(self, full_expert=True):
        if full_expert:
            model_path = "experts/models/rgb2normal_consistency.pth"
            self.model = UNet()
            model_state_dict = torch.load(model_path)
            self.model.load_state_dict(model_state_dict)
            self.model.eval()

            self.trans_totensor = transforms.Compose([
                transforms.Resize(W * 2, interpolation=PIL.Image.BILINEAR),
                transforms.CenterCrop(W),
                transforms.ToTensor()
            ])
        self.domain_name = "surface_normals"
        self.n_maps = 3
        self.str_id = "xtc_surface_normals"

    def apply_expert(self, rgb_frames):
        normals_maps = []
        for idx, rgb_frame in enumerate(rgb_frames):
            img_tensor = self.trans_totensor(rgb_frame)[:3].unsqueeze(0)
            output_map = self.model(img_tensor).clamp(min=0, max=1).data.cpu()
            output_map = output_map.permute(0, 2, 3, 1)
            normals_maps.append(output_map)

        # save_fname = "normals_test.png"
        # print("Save Surface Normals to %s" % save_fname)
        # cv2.imwrite(save_fname, output_map[0].numpy() * 255.)

        return torch.cat(normals_maps, dim=0)

    def apply_expert_one_frame(self, rgb_frame):
        img_tensor = self.trans_totensor(rgb_frame)[:3].unsqueeze(0)
        output_map = self.model(img_tensor).clamp(min=0, max=1).data.cpu()[0]
        return output_map
