import cv2
import numpy as np 
import torch 
import torchvision
from torchvision import models 

class DeepLabv3Test:

    def __init__(self, full_expert=True):
        if full_expert:
            self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
            self.model.cuda()
            self.model.eval()
        self.domain_name = "semantic_segmentation"
        self.n_maps = 21
        self.str_id = 'sseg_deeplabv3'
        
    def apply(self, frame):
        # frame should be RGB 
        imagenet_mean =torch.tensor([0.485, 0.456, 0.406])
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
        result = result[0,:,:,:]
        result = torch.softmax(result, 0)
        result = result.permute(1,2,0)
        result = result.cpu()

        return result

    def apply_expert(self, frames):
        # frame should be RGB 
        imagenet_mean =torch.tensor([0.485, 0.456, 0.406])
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
            result = result[0,:,:,:]
            result = torch.softmax(result, 0)
            #result = result.permute(1,2,0)
            result = result.cpu()

            results.append(result)

        return results