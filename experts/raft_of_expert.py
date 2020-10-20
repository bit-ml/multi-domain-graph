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

class RaftTest:

    def __init__(self, model_path):
        #import pdb 
        #pdb.set_trace()
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', help="restore checkpoint")
        parser.add_argument('--path', help="dataset for evaluation")
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
        str1 = '--model=%s'%model_path
        args = parser.parse_args([str1])
        #args = parser.parse_args(['--model=raft_optical_flow/models/raft-things.pth'])

        self.model = torch.nn.DataParallel(RAFT(args))
        self.model.load_state_dict(torch.load(args.model))

        self.model = self.model.module
        self.model.to(DEVICE)
        self.model.eval()

    def apply(self, img1, img2):
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float().cuda()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float().cuda()
        img1 = img1[None]
        img2 = img2[None]

        with torch.no_grad():
            padder = InputPadder(img1.shape)
            img1, img2 = padder.pad(img1, img2)

            _, flow_up = self.model(img1, img2, iters=20, test_mode=True)

        return flow_up[0,:,:,:]