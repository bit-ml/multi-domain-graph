import torch

import math
import numpy
import os
import sys

import experts.liteflownet_optical_flow.run

class LiteFlowNetTest:
    def __init__(self, model_path):
        self.netNetwork = experts.liteflownet_optical_flow.run.Network(model_path).cuda().eval()


    def apply(self, img1, img2):   
        # img1, img2 - numpy arrays

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


