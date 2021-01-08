import torch

import math
import numpy
import os
import sys
import numpy as np
import experts.liteflownet_optical_flow.run

#LITE_FLOW_NET_MODEL_PATH = r'experts/liteflownet_optical_flow/models/liteflownet-default'
current_dir_name = os.path.dirname(os.path.realpath(__file__))
LITE_FLOW_NET_MODEL_PATH = os.path.join(current_dir_name,
                                        'models/of_liteflownet')


class LiteFlowNetModel:
    def __init__(self, full_expert=True, fwd=1):
        if full_expert:
            self.netNetwork = experts.liteflownet_optical_flow.run.Network(
                LITE_FLOW_NET_MODEL_PATH).cuda().eval()

        self.fwd = fwd
        self.n_maps = 2
        if fwd:
            self.domain_name = 'of_fwd'
        else:
            self.domain_name = 'of_bwd'
        self.str_id = "liteflownet"
        self.identifier = self.domain_name + "_" + self.str_id

    def aux(self, tenFirst, tenSecond):

        assert (tenFirst.shape[1] == tenSecond.shape[1])
        assert (tenFirst.shape[2] == tenSecond.shape[2])

        intWidth = tenFirst.shape[2]
        intHeight = tenFirst.shape[1]

        tenPreprocessedFirst = tenFirst.view(1, 3, intHeight, intWidth)
        tenPreprocessedSecond = tenSecond.view(1, 3, intHeight, intWidth)
        '''
        tenFirst = torch.FloatTensor(
            numpy.ascontiguousarray(
                img1[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) *
                (1.0 / 255.0)))
        tenSecond = torch.FloatTensor(
            numpy.ascontiguousarray(
                img2[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) *
                (1.0 / 255.0)))
        assert (tenFirst.shape[1] == tenSecond.shape[1])
        assert (tenFirst.shape[2] == tenSecond.shape[2])

        intWidth = tenFirst.shape[2]
        intHeight = tenFirst.shape[1]

        tenPreprocessedFirst = tenFirst.cuda().view(1, 3, intHeight, intWidth)
        tenPreprocessedSecond = tenSecond.cuda().view(1, 3, intHeight,
                                                      intWidth)
        '''
        intPreprocessedWidth = int(
            math.floor(math.ceil(intWidth / 32.0) * 32.0))
        intPreprocessedHeight = int(
            math.floor(math.ceil(intHeight / 32.0) * 32.0))

        tenPreprocessedFirst = torch.nn.functional.interpolate(
            input=tenPreprocessedFirst,
            size=(intPreprocessedHeight, intPreprocessedWidth),
            mode='bilinear',
            align_corners=False)
        tenPreprocessedSecond = torch.nn.functional.interpolate(
            input=tenPreprocessedSecond,
            size=(intPreprocessedHeight, intPreprocessedWidth),
            mode='bilinear',
            align_corners=False)

        tenFlow = torch.nn.functional.interpolate(input=self.netNetwork(
            tenPreprocessedFirst, tenPreprocessedSecond),
                                                  size=(intHeight, intWidth),
                                                  mode='bilinear',
                                                  align_corners=False)

        tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

        return tenFlow[0, :, :, :].cpu()

    def apply_expert_batch(self, batch_rgb_frames):
        if self.fwd == 1:
            batch_rgb_frames = torch.cat(
                (batch_rgb_frames, batch_rgb_frames[-1, :, :, :][None]), 0)
        else:
            batch_rgb_frames = torch.cat(
                (batch_rgb_frames[0, :, :, :][None], batch_rgb_frames), 0)
        batch_rgb_frames = batch_rgb_frames.numpy()
        batch_rgb_frames = torch.FloatTensor(
            numpy.ascontiguousarray(batch_rgb_frames[:, :, :, ::-1].transpose(
                0, 3, 1, 2).astype(numpy.float32) * (1.0 / 255.0)))
        batch_rgb_frames = batch_rgb_frames.cuda()

        flows = []
        if self.fwd == 1:
            for idx in range(batch_rgb_frames.shape[0] - 1):
                flows.append(
                    self.aux(batch_rgb_frames[idx, :, :, :],
                             batch_rgb_frames[idx + 1, :, :, :]).numpy())
        else:
            for idx in range(batch_rgb_frames.shape[0] - 1):
                flows.append(
                    self.aux(batch_rgb_frames[idx + 1, :, :, :],
                             batch_rgb_frames[idx, :, :, :]).numpy())

        flows = np.array(flows).astype('float32')
        return flows

    '''
    def apply(self, img1, img2):
        if self.fwd == 1:
            return self.aux(img1, img2)
        else:
            return self.aux(img2, img1)

    def apply_expert(self, frames):
        flows = []
        if self.fwd == 1:
            prev_img = frames[0]
            for i in range(len(frames) - 1):
                current_img = frames[i + 1]
                flows.append(self.aux(prev_img, current_img))
                prev_img = current_img
            flows.append(self.aux(prev_img, prev_img))
        else:
            prev_img = frames[0]
            flows.append(self.aux(prev_img, prev_img))
            for i in range(len(frames) - 1):
                current_img = frames[i + 1]
                flows.append(self.aux(current_img, prev_img))
                prev_img = current_img
        return flows
    '''