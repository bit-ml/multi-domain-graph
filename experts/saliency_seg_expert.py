# use SGDepth code for depth expert - https://github.com/xavysp/DexiNed/blob/master/DexiNed-Pytorch/
import os

import cv2
import numpy as np
import tensorflow as tf
import torch
from torchvision import transforms

from experts.basic_expert import BasicExpert
from experts.saliencysegm.model import build_model, weights_init

W, H = 256, 256

current_dir_name = os.path.dirname(os.path.realpath(__file__))
saliency_model_path = os.path.join(current_dir_name,
                                   'models/saliency_seg_egnet.pth')


class SaliencySegmModel(BasicExpert):
    def __init__(self, full_expert=True):
        if full_expert:
            #resnet_path = 'experts/models/resnet50_caffe.pth'
            checkpoint_path = saliency_model_path  #"experts/models/egnet_resnet.pth"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device

            self.mean = torch.Tensor([123.68, 116.779, 103.939]).view(3, 1,
                                                                      1) / 255.
            self.net_bone = build_model().to(device)
            # self.net_bone.base.load_state_dict(torch.load(resnet_path))
            self.net_bone.load_state_dict(torch.load(checkpoint_path))
            self.net_bone.eval()
            self.trans_totensor = transforms.Compose([transforms.ToTensor()])

        self.domain_name = "saliency_seg"
        self.n_maps = 1

        self.str_id = "egnet"
        self.identifier = self.domain_name + "_" + self.str_id

    def apply_expert_batch(self, batch_rgb_frames):
        edge_maps = []
        for idx, rgb_frame in enumerate(batch_rgb_frames):
            rgb_frame = rgb_frame.numpy()
            rgb_frame = rgb_frame.astype('float32')
            input_tensor = self.trans_totensor(rgb_frame)[None].to(self.device)
            up_edge, up_sal, up_sal_f = self.net_bone(input_tensor)
            pred = np.squeeze(torch.sigmoid(up_sal_f[-1]).cpu().data.numpy())
            edge_maps.append(pred[None])
        edge_maps = np.array(edge_maps).astype('float32')
        return edge_maps

    '''
    def apply_expert(self, rgb_frames):
        edge_maps = []

        for rgb_frame in rgb_frames:
            resized_rgb_frame = cv2.resize(np.array(rgb_frame),
                                           (W, H)).astype(np.float32)
            input_tensor = self.trans_totensor(resized_rgb_frame)[None].to(
                self.device)
            up_edge, up_sal, up_sal_f = self.net_bone(input_tensor)
            pred = np.squeeze(torch.sigmoid(up_sal_f[-1]).cpu().data.numpy())
            edge_maps.append(pred[None])
        return edge_maps

    def apply_expert_one_frame(self, rgb_frame):
        resized_rgb_frame = cv2.resize(np.array(rgb_frame),
                                       (W, H)).astype(np.float32)
        input_tensor = self.trans_totensor(resized_rgb_frame)[None].to(
            self.device)
        up_edge, up_sal, up_sal_f = self.net_bone(input_tensor)
        pred = np.squeeze(torch.sigmoid(up_sal_f[-1]).cpu().data.numpy())

        # out shape: expert.n_maps x 256 x 256
        return pred[None]
    '''
