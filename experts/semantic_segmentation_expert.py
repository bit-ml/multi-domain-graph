import os

import encoding
import numpy as np
import torch
import torchvision
from skimage import color
from torch import nn
from torchvision import models

from experts.basic_expert import BasicExpert
from experts.semantic_segmentation.hrnet.mit_semseg.models import ModelBuilder

current_dir_name = os.path.dirname(os.path.realpath(__file__))
ss_model_path = os.path.join(current_dir_name, 'models/')


# ADE20k labels https://github.com/CSAILVision/sceneparsing/blob/master/objectInfo150.csv
class SSegHRNet(BasicExpert):
    def __init__(self, full_expert=True):
        if full_expert:
            self.num_class = 150
            # Network Builders
            self.encoder = ModelBuilder.build_encoder(
                arch="hrnetv2",
                fc_dim=720,
                weights="%s/hrnet_encoder_epoch_30.pth" % ss_model_path)
            self.decoder = ModelBuilder.build_decoder(
                arch="c1",
                fc_dim=720,
                num_class=self.num_class,
                weights="%s/hrnet_decoder_epoch_30.pth" % ss_model_path,
                use_softmax=True)

            self.encoder.eval()
            self.decoder.eval()

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            self.encoder.to(device)
            self.decoder.to(device)

        self.domain_name = "sem_seg"
        self.n_maps = 1
        self.str_id = "hrnet"
        self.identifier = self.domain_name + "_" + self.str_id

    def apply_expert_batch(self, batch_rgb_frames):
        batch_rgb_frames = batch_rgb_frames.permute(0, 3, 1, 2) / 255.
        sseg_maps = self.decoder(self.encoder(batch_rgb_frames.to(self.device),
                                              return_feature_maps=True),
                                 segSize=256)

        sseg_maps = self.post_process_ops(sseg_maps, self.expert_specific)

        sseg_maps = np.array(sseg_maps.data.cpu().numpy()).astype('float32')
        return sseg_maps

    def post_process_ops(self, pred_logits, specific_fcn):
        sseg_maps = specific_fcn(pred_logits)
        return sseg_maps

    def expert_specific(self, inp):
        '''
        inp shape NCHW, ex: torch.Size([30, 150, 256, 256])
        return one single map with class_labels
        '''
        class_labels = inp.argmax(dim=1)[:, None] / self.num_class

        return class_labels

    def edge_specific(self, inp):
        outp = inp.round().clamp(min=0, max=1)
        return outp


class SSegResNeSt(BasicExpert):
    def __init__(self, full_expert=True):
        if full_expert:
            model_name = "EncNet_ResNet50s_ADE"
            self.model = encoding.models.get_model(model_name, pretrained=True)
            self.model.eval()

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            self.model.to(device)

        self.domain_name = "sem_seg"
        self.n_maps = 1
        self.str_id = "resnest"
        self.identifier = self.domain_name + "_" + self.str_id

    def apply_expert_batch(self, batch_rgb_frames):
        batch_rgb_frames = batch_rgb_frames.permute(0, 3, 1, 2) / 255.
        sseg_maps = self.model(batch_rgb_frames.to(self.device))
        sseg_maps = sseg_maps.clamp(min=0, max=1).data.cpu().numpy()
        sseg_maps = np.array(sseg_maps).astype('float32')
        return sseg_maps


class DeepLabv3Model(BasicExpert):
    def __init__(self, full_expert=True):
        if full_expert:
            self.model = torchvision.models.segmentation.deeplabv3_resnet101(
                pretrained=True)
            self.model.cuda()
            self.model.eval()
        self.domain_name = "sem_seg"
        self.n_maps = 1
        self.str_id = 'deeplabv3'
        self.identifier = self.domain_name + "_" + self.str_id

    def apply_expert_batch(self, batch_rgb_frames):
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406])[None, None,
                                                            None, :]
        imagenet_std = torch.tensor([0.229, 0.224, 0.225])[None, None, None, :]

        batch_rgb_frames = batch_rgb_frames.float() / 255.
        batch_rgb_frames = (batch_rgb_frames - imagenet_mean) / imagenet_std

        batch_rgb_frames = batch_rgb_frames.permute(0, 3, 1, 2).cuda()
        results = self.model(batch_rgb_frames)
        results = results['out'].detach()
        results = torch.softmax(results, 1)
        results = results.cpu().numpy().astype('float32')
        return results

    '''
    def apply(self, frame):
        # frame should be RGB
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
        imagenet_std = torch.tensor([0.229, 0.224, 0.225])

        frame = torch.tensor(frame)
        frame = frame.float()
        frame = frame / 255

        imagenet_mean = imagenet_mean[None, None, :]
        imagenet_std = imagenet_std[None, None, :]
        frame = (frame - imagenet_mean) / imagenet_std

        frame = frame.permute(2, 0, 1)

        frame = frame.cuda()
        result = self.model(frame.unsqueeze(0))
        result = result['out']
        result = result.detach()
        result = result[0, :, :, :]
        result = torch.softmax(result, 0)
        result = result.permute(1, 2, 0)
        result = result.cpu()

        return result

    def apply_expert(self, frames):
        # frame should be RGB
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
        imagenet_std = torch.tensor([0.229, 0.224, 0.225])
        imagenet_mean = imagenet_mean[None, None, :]
        imagenet_std = imagenet_std[None, None, :]

        results = []
        for i in range(len(frames)):

            frame = torch.tensor(frames[i])
            frame = frame.float()
            frame = frame / 255
            frame = (frame - imagenet_mean) / imagenet_std

            frame = frame.permute(2, 0, 1)

            frame = frame.cuda()
            result = self.model(frame.unsqueeze(0))
            result = result['out']
            result = result.detach()
            result = result[0, :, :, :]
            result = torch.softmax(result, 0)
            #result = result.permute(1,2,0)
            result = result.cpu().numpy()

            results.append(result)

        return results
    '''


class FCNModel(BasicExpert):
    def __init__(self, full_expert=True):
        if full_expert:
            self.model = torchvision.models.segmentation.fcn_resnet101(
                pretrained=True)
            self.model.cuda()
            self.model.eval()
        self.domain_name = "sem_seg"
        self.n_maps = 21
        self.str_id = 'fcn'
        self.identifier = self.str_id

    def apply_expert_batch(self, batch_rgb_frames):
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406])[None, None,
                                                            None, :]
        imagenet_std = torch.tensor([0.229, 0.224, 0.225])[None, None, None, :]

        batch_rgb_frames = batch_rgb_frames.float() / 255.
        batch_rgb_frames = (batch_rgb_frames - imagenet_mean) / imagenet_std

        batch_rgb_frames = batch_rgb_frames.permute(0, 3, 1, 2).cuda()
        results = self.model(batch_rgb_frames)
        results = results['out'].detach()
        results = torch.softmax(results, 1)
        results = results.cpu().numpy().astype('float32')
        return results

    '''
        def apply(self, frame):
        # frame should be RGB
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
        imagenet_std = torch.tensor([0.229, 0.224, 0.225])

        frame = torch.tensor(frame)
        frame = frame.float()
        frame = frame / 255

        imagenet_mean = imagenet_mean[None, None, :]
        imagenet_std = imagenet_std[None, None, :]
        frame = (frame - imagenet_mean) / imagenet_std

        frame = frame.permute(2, 0, 1)

        frame = frame.cuda()
        result = self.model(frame.unsqueeze(0))
        result = result['out']
        result = result.detach()
        result = result[0, :, :, :]
        result = torch.softmax(result, 0)
        result = result.permute(1, 2, 0)
        result = result.cpu()

        return result
    def apply_expert(self, frames):
        # frame should be RGB
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406])[None, None, :]
        imagenet_std = torch.tensor([0.229, 0.224, 0.225])[None, None, :]

        results = []
        for i in range(len(frames)):

            frame = torch.tensor(frames[i])
            frame = frame.float()
            frame = frame / 255
            frame = (frame - imagenet_mean) / imagenet_std

            frame = frame.permute(2, 0, 1)

            frame = frame.cuda()
            result = self.model(frame.unsqueeze(0))
            result = result['out']
            result = result.detach()
            result = result[0, :, :, :]
            result = torch.softmax(result, 0)
            #result = result.permute(1,2,0)
            result = result.cpu().numpy()

            results.append(result)

        return results
    '''
