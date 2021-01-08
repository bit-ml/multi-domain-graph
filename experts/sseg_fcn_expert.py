import cv2
import numpy as np
import torch
import torchvision
from torchvision import models


class FCNModel:
    def __init__(self, full_expert=True):
        if full_expert:
            self.model = torchvision.models.segmentation.fcn_resnet101(
                pretrained=True)
            self.model.cuda()
            self.model.eval()
        self.domain_name = "semantic_segmentation"
        self.n_maps = 21
        self.str_id = 'sseg_fcn'
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