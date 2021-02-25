# https://github.com/EPFL-VILAB/XTConsistency
import os

import cv2
import numpy as np
import PIL
import torch
from tensorflow.python.keras.utils.generic_utils import \
    _SKIP_FAILED_SERIALIZATION
from torchvision import transforms

from experts.basic_expert import BasicExpert
from experts.xtc.unet import UNet

W, H = 256, 256

current_dir_name = os.path.dirname(os.path.realpath(__file__))
normals_model_path = os.path.join(current_dir_name, 'models/normals_xtc.pth')


class SurfaceNormalsXTC(BasicExpert):
    SOME_THRESHOLD = 0.

    def __init__(self, dataset_name, full_expert=True):
        '''
            dataset_name: "taskonomy" or "replica"
        '''
        if full_expert:
            model_path = normals_model_path
            self.model = UNet()
            model_state_dict = torch.load(model_path)
            self.model.load_state_dict(model_state_dict)
            self.model.eval()
            self.dataset_name = dataset_name
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            self.model.to(device)

        self.domain_name = "normals"
        self.n_final_maps = 3

        if dataset_name == "taskonomy":
            # self.chan_replace = 0
            # self.chan_gen_fcn = torch.zeros_like
            self.edge_specific = lambda x: x
            self.expert_specific = lambda x: x
            self.n_maps = 3
        else:
            self.chan_replace = 0
            self.chan_gen_fcn = torch.ones_like
            self.n_maps = 2
            # self.edge_specific = lambda x: x
            # self.expert_specific = lambda x: x
            # self.n_maps = 3

        self.str_id = "xtc"
        self.identifier = self.domain_name + "_" + self.str_id

    def apply_expert_batch(self, batch_rgb_frames):
        batch_rgb_frames = batch_rgb_frames.permute(0, 3, 1, 2) / 255.
        out_maps = self.model(batch_rgb_frames.to(self.device))

        # 1. CLAMP
        torch.clamp_(out_maps[:, :2], min=0, max=1)
        torch.clamp_(out_maps[:, 2], min=0., max=0.5)

        # 2. ALIGN ranges
        out_maps[:, 2] += 0.5

        # 4. NORMALIZE it
        out_maps = out_maps * 2 - 1
        out_maps[:, 2] = SurfaceNormalsXTC.SOME_THRESHOLD
        norm_normals_maps = torch.norm(out_maps, dim=1, keepdim=True)
        norm_normals_maps[norm_normals_maps == 0] = 1
        out_maps = out_maps / norm_normals_maps
        out_maps = (out_maps + 1) / 2

        out_maps = out_maps.data.cpu().numpy().astype('float32')

        return out_maps

    def no_maps_as_nn_input(self):
        return self.n_final_maps

    def no_maps_as_nn_output(self):
        return self.n_maps

    def no_maps_as_ens_input(self):
        return self.n_final_maps

    def postprocess_train(self, nn_outp):
        return nn_outp

    def postprocess_eval(self, nn_outp):
        # add the 3rd dimension
        nn_outp = torch.cat(
            (nn_outp, self.chan_gen_fcn(nn_outp[:, 1][:, None]) / 2), dim=1)

        # 1. CLAMP
        torch.clamp_(nn_outp[:, :2], min=0, max=1)

        # 4. NORMALIZE it
        nn_outp = nn_outp * 2 - 1
        nn_outp[:, 2] = SurfaceNormalsXTC.SOME_THRESHOLD
        norm_normals_maps = torch.norm(nn_outp, dim=1, keepdim=True)
        norm_normals_maps[norm_normals_maps == 0] = 1
        nn_outp = nn_outp / norm_normals_maps
        nn_outp = (nn_outp + 1) / 2

        return nn_outp

    def postprocess_eval_ens(self, nn_outp):
        # 1. CLAMP
        torch.clamp_(nn_outp[:, :2], min=0, max=1)

        # 4. NORMALIZE it
        nn_outp = nn_outp * 2 - 1
        nn_outp[:, 2] = SurfaceNormalsXTC.SOME_THRESHOLD
        norm_normals_maps = torch.norm(nn_outp, dim=1, keepdim=True)
        norm_normals_maps[norm_normals_maps == 0] = 1
        nn_outp = nn_outp / norm_normals_maps
        nn_outp = (nn_outp + 1) / 2

        return nn_outp

    def gt_train_transform(edge, gt_inp):
        return gt_inp[:, :2]

    # to_ens_transform = (lambda x, y: x)
