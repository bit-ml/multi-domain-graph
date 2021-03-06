import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from experts.raft_optical_flow.core.raft import RAFT
from experts.raft_optical_flow.core.raft_utils import InputPadder

DEVICE = 'cuda'

#RAFT_MODEL_PATH = r'experts/raft_optical_flow/models/raft-kitti.pth'
current_dir_name = os.path.dirname(os.path.realpath(__file__))
RAFT_MODEL_PATH = os.path.join(current_dir_name, 'models/of_raft.pth')


class RaftModel:
    def __init__(self, full_expert=True, fwd=1):
        if full_expert:
            #import pdb
            #pdb.set_trace()
            parser = argparse.ArgumentParser()
            parser.add_argument('--model', help="restore checkpoint")
            parser.add_argument('--path', help="dataset for evaluation")
            parser.add_argument('--small',
                                action='store_true',
                                help='use small model')
            parser.add_argument('--mixed_precision',
                                action='store_true',
                                help='use mixed precision')
            parser.add_argument('--alternate_corr',
                                action='store_true',
                                help='use efficent correlation implementation')
            str1 = '--model=%s' % RAFT_MODEL_PATH
            args = parser.parse_args([str1])
            #args = parser.parse_args(['--model=raft_optical_flow/models/raft-things.pth'])

            self.model = torch.nn.DataParallel(RAFT(args))
            self.model.load_state_dict(torch.load(args.model))

            self.model = self.model.module
            self.model.to(DEVICE)
            self.model.eval()

        self.fwd = fwd

        self.n_maps = 2
        if fwd:
            self.domain_name = "of_fwd"
        else:
            self.domain_name = "of_bwd"
        self.str_id = "raft"
        self.identifier = self.domain_name + "_" + self.str_id

    def apply_expert_batch(self, batch_rgb_frames):
        if self.fwd == 1:
            batch_rgb_frames = torch.cat(
                (batch_rgb_frames, batch_rgb_frames[-1, :, :, :][None]), 0)
        else:
            batch_rgb_frames = torch.cat(
                (batch_rgb_frames[0, :, :, :][None], batch_rgb_frames), 0)

        batch_rgb_frames = batch_rgb_frames.permute(0, 3, 1, 2).float().cuda()

        if self.fwd == 1:
            flows = []
            for idx in range(batch_rgb_frames.shape[0] - 1):
                prev_img = batch_rgb_frames[idx, :, :, :][None]
                current_img = batch_rgb_frames[idx + 1, :, :, :][None]

                with torch.no_grad():
                    padder = InputPadder(prev_img.shape)
                    img1, img2 = padder.pad(prev_img.clone(),
                                            current_img.clone())
                    _, flow_up = self.model(img1,
                                            img2,
                                            iters=20,
                                            test_mode=True)
                    flows.append(flow_up[0, :, :, :].cpu().numpy())
        else:
            flows = []
            for idx in range(batch_rgb_frames.shape[0] - 1):
                prev_img = batch_rgb_frames[idx, :, :, :][None]
                current_img = batch_rgb_frames[idx + 1, :, :, :][None]

                with torch.no_grad():
                    padder = InputPadder(prev_img.shape)
                    img1, img2 = padder.pad(prev_img.clone(),
                                            current_img.clone())
                    _, flow_up = self.model(img2,
                                            img1,
                                            iters=20,
                                            test_mode=True)
                    flows.append(flow_up[0, :, :, :].cpu().numpy())

        flows = np.array(flows).astype('float32')
        return flows

    '''
    def apply(self, img1, img2):
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float().cuda()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float().cuda()
        img1 = img1[None]
        img2 = img2[None]

        with torch.no_grad():
            padder = InputPadder(img1.shape)
            img1, img2 = padder.pad(img1, img2)
            if self.fwd == 1:
                _, flow_up = self.model(img1, img2, iters=20, test_mode=True)
            else:
                _, flow_up = self.model(img2, img1, iters=20, test_mode=True)

        return flow_up[0, :, :, :]

    def apply_expert(self, frames):
        flows = []
        if self.fwd == 1:
            prev_img = frames[0]
            prev_img = torch.from_numpy(prev_img).permute(2, 0,
                                                          1).float().cuda()
            prev_img = prev_img[None]

            for i in range(len(frames) - 1):
                current_img = frames[i + 1]
                current_img = torch.from_numpy(current_img).permute(
                    2, 0, 1).float().cuda()
                current_img = current_img[None]

                with torch.no_grad():
                    padder = InputPadder(prev_img.shape)
                    img1, img2 = padder.pad(prev_img.clone(),
                                            current_img.clone())
                    _, flow_up = self.model(img1,
                                            img2,
                                            iters=20,
                                            test_mode=True)
                    flows.append(flow_up[0, :, :, :].cpu().numpy())

                prev_img = current_img.clone()
            with torch.no_grad():
                padder = InputPadder(prev_img.shape)
                img1, img2 = padder.pad(prev_img.clone(), prev_img.clone())
                _, flow_up = self.model(img1, img2, iters=20, test_mode=True)
                flows.append(flow_up[0, :, :, :].cpu().numpy())

        else:
            prev_img = frames[0]
            prev_img = torch.from_numpy(prev_img).permute(2, 0,
                                                          1).float().cuda()
            prev_img = prev_img[None]
            with torch.no_grad():
                padder = InputPadder(prev_img.shape)
                img1, img2 = padder.pad(prev_img.clone(), prev_img.clone())
                _, flow_up = self.model(img2, img1, iters=20, test_mode=True)
                flows.append(flow_up[0, :, :, :].cpu().numpy())

            for i in range(len(frames) - 1):
                current_img = frames[i + 1]
                current_img = torch.from_numpy(current_img).permute(
                    2, 0, 1).float().cuda()
                current_img = current_img[None]

                with torch.no_grad():
                    padder = InputPadder(prev_img.shape)
                    img1, img2 = padder.pad(prev_img.clone(),
                                            current_img.clone())
                    _, flow_up = self.model(img2,
                                            img1,
                                            iters=20,
                                            test_mode=True)
                    flows.append(flow_up[0, :, :, :].cpu().numpy())

                prev_img = current_img.clone()
        return flows
    '''