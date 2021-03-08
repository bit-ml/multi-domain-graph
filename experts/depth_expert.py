# use SGDepth code for depth expert - https://github.com/ifnspaml/SGDepth
import os

import numpy as np
import torch
from torchvision import transforms

from experts.basic_expert import BasicExpert
from experts.depth.arguments import InferenceEvaluationArguments
from experts.depth.inference import Inference
from experts.xtc.unet import UNet

W, H = 256, 256

current_dir_name = os.path.dirname(os.path.realpath(__file__))
depth_model_path = os.path.join(current_dir_name, 'models/depth_sgdepth.pth')
depthxtc_model_path = os.path.join(current_dir_name, 'models/depth_xtc.pth')


class DepthModel():
    def __init__(self, full_expert=True):
        if full_expert:
            opt = InferenceEvaluationArguments().parse()
            opt.model_path = depth_model_path  #"experts/models/depth_sgdepth.pth"
            # the model is trained for this inference size!!
            opt.inference_resize_height = H
            opt.inference_resize_width = W
            self.model = Inference(opt)
            self.model.model.eval()
        self.domain_name = "depth_n_1"
        self.n_maps = 1
        self.str_id = "sgdepth"
        self.identifier = self.domain_name + "_" + self.str_id

    def apply_expert_batch(self, batch_rgb_frames):
        batch_rgb_frames = batch_rgb_frames / 255.
        depth_maps = []
        for idx, rgb_frame in enumerate(batch_rgb_frames):
            depth_map, segm_map = self.model.inference(rgb_frame.numpy())
            depth_maps.append(depth_map[0].cpu().numpy())
        depth_maps = np.array(depth_maps).astype('float32')

        return 1 / (depth_maps)  #1 - depth_maps

    '''
    def apply_expert(self, rgb_frames):
        depth_maps = []
        for idx, rgb_frame in enumerate(rgb_frames):
            depth_map, segm_map = self.model.inference(rgb_frame)

            #save_fname = "depth_test.png"
            #print("Save depth in %s" % save_fname)
            #self.model.save_pred_to_disk(depth_map, segm_map, save_fname)

            depth_maps.append(depth_map[0].cpu().numpy())
        return depth_maps  #torch.cat(depth_maps, dim=0).cpu().numpy()

    def apply_expert_one_frame(self, rgb_frame):
        depth_map, segm_map = self.model.inference(rgb_frame)
        depth_map = depth_map.data.cpu()[0]
        return depth_map
    '''


class DepthModelXTC(BasicExpert):
    def __init__(self, full_expert=True):
        if full_expert:
            model_path = depthxtc_model_path
            self.model = UNet(downsample=6, out_channels=1)
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
        self.domain_name = "depth_n_1"
        self.n_maps = 1
        self.str_id = "xtc"
        self.identifier = self.domain_name + "_" + self.str_id

    # def apply_expert_batch(self, batch_rgb_frames):
    #     depth_maps = []
    #     for idx, rgb_frame in enumerate(batch_rgb_frames):
    #         depth_map, segm_map = self.model(rgb_frame.numpy())
    #         depth_maps.append(depth_map[0].cpu().numpy())
    #     depth_maps = np.array(depth_maps).astype('float32')
    #     return depth_maps

    def apply_expert_batch(self, batch_rgb_frames):
        batch_rgb_frames = batch_rgb_frames.permute(0, 3, 1, 2) / 255.
        depth_maps = self.model(batch_rgb_frames.to(self.device))
        depth_maps = depth_maps.data.cpu().numpy()
        #depth_maps = depth_maps.clamp(min=0, max=1).data.cpu().numpy()
        depth_maps = np.array(depth_maps).astype('float32')
        return depth_maps

    def test_gt(self, loss_fct, pred, target):
        l_target = target.clone()
        l_pred = pred.clone()
        bm = l_target != l_target
        bm = ~bm
        l_target[l_target != l_target] = 0
        l_target = l_target * bm
        l_pred = l_pred * bm
        loss = loss_fct(l_pred, l_target)
        return loss
