import torch

import math
import numpy
import os
import sys

import experts.liteflownet_optical_flow.run

class LiteFlowNetTest:
    def __init__(self, model_path, fwd):
        self.netNetwork = experts.liteflownet_optical_flow.run.Network(model_path).cuda().eval()
        self.fwd = fwd

    def aux(self, img1, img2):
        tenFirst = torch.FloatTensor(numpy.ascontiguousarray(img1[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
        tenSecond = torch.FloatTensor(numpy.ascontiguousarray(img2[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
        assert(tenFirst.shape[1] == tenSecond.shape[1])
        assert(tenFirst.shape[2] == tenSecond.shape[2])

        intWidth = tenFirst.shape[2]
        intHeight = tenFirst.shape[1]

        tenPreprocessedFirst = tenFirst.cuda().view(1, 3, intHeight, intWidth)
        tenPreprocessedSecond = tenSecond.cuda().view(1, 3, intHeight, intWidth)

        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

        tenPreprocessedFirst = torch.nn.functional.interpolate(input=tenPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
        tenPreprocessedSecond = torch.nn.functional.interpolate(input=tenPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

        tenFlow = torch.nn.functional.interpolate(input=self.netNetwork(tenPreprocessedFirst, tenPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

        tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

        return tenFlow[0, :, :, :].cpu()

    def apply(self, img1, img2):   
        if self.fwd==1:
            return self.aux(img1, img2)
        else:
            return self.aux(img2, img1)

    def apply_per_video(self, frames):   
        flows = []
        if self.fwd==1:
            prev_img = frames[0]
            for i in range(len(frames)-1):
                current_img = frames[i+1]
                flows.append(self.aux(prev_img, current_img))
                prev_img = current_img
            flows.append(self.aux(prev_img, prev_img))
        else:
            prev_img = frames[0]
            flows.append(self.aux(prev_img, prev_img))
            for i in range(len(frames)-1):
                current_img = frames[i+1]
                flows.append(self.aux(current_img, prev_img))
                prev_img = current_img
        return flows
