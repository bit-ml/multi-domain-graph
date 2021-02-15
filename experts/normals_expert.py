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
    def __init__(self, dataset_name, full_expert=True):
        '''
            dataset_name: "taskonomy" or "replica"
        '''
        if full_expert:
            model_path = normals_model_path  #"experts/models/rgb2normal_consistency.pth"
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
            self.chan_replace = 1
            self.chan_gen_fcn = torch.ones_like
            self.n_maps = 2

        self.str_id = "xtc"
        self.identifier = self.domain_name + "_" + self.str_id

    def apply_expert_batch(self, batch_rgb_frames):
        batch_rgb_frames = batch_rgb_frames.permute(0, 3, 1, 2) / 255.
        normals_maps = self.model(batch_rgb_frames.to(self.device))

        #normals_maps = self.post_process_ops(normals_maps,
        #                                     self.expert_specific)

        normals_maps = normals_maps.data.cpu().numpy().astype('float32')
        return normals_maps

    def post_process_ops(self, pred_logits, specific_fcn):

        pred_logits = torch.clamp(pred_logits, 0, 1)
        pred_logits = pred_logits * 2 - 1
        norm_pred_logits = torch.norm(pred_logits, dim=1, keepdim=True)

        pred_logits = pred_logits / norm_pred_logits
        pred_logits = (pred_logits + 1) / 2

        return pred_logits

    def expert_specific(self, inp):
        #inp[:, 2, :, :] = self.chan_replace
        return inp

    def edge_specific_train(self, inp):
        inp = torch.cat((inp, self.chan_gen_fcn(inp[:, 1][:, None])), dim=1)
        return inp

    def edge_specific_eval(self, inp):
        inp = torch.cat((inp, self.chan_gen_fcn(inp[:, 1][:, None])), dim=1)
        return inp

    def no_maps_as_input(self):
        return 3

    def no_maps_as_output(self):
        return self.n_maps
