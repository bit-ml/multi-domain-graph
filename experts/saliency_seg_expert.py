# use SGDepth code for depth expert - https://github.com/xavysp/DexiNed/blob/master/DexiNed-Pytorch/

import cv2
import numpy as np
import tensorflow as tf
import torch
from torchvision import transforms

from experts.saliencysegm.model import build_model, weights_init

W, H = 256, 256


class SaliencySegmModel():
    def __init__(self, full_expert=True):
        if full_expert:
            resnet_path = 'experts/models/resnet50_caffe.pth'
            checkpoint_path = "experts/models/egnet_resnet.pth"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device

            self.mean = torch.Tensor([123.68, 116.779, 103.939]).view(3, 1, 1) / 255.
            self.net_bone = build_model().to(device)
            # self.net_bone.base.load_state_dict(torch.load(resnet_path))
            self.net_bone.load_state_dict(torch.load(checkpoint_path))
            self.net_bone.eval()
            self.trans_totensor = transforms.Compose([
                transforms.ToTensor()
            ])

        self.domain_name = "salientseg"
        self.n_maps = 1
        self.str_id = "egnet"

    # def apply_expert(self, rgb_frames):
    #     edge_maps = []
    #     for idx, rgb_frame in enumerate(rgb_frames):
    #         resized_rgb_frame = cv2.resize(np.array(rgb_frame),
    #                                        (W, H)).astype(np.float32)
    #         preds = self.model(resized_rgb_frame, training=False)
    #         edge_map = tf.sigmoid(preds).numpy()[:, :, :, 0]

    #         save_fname = "edge_test.png"
    #         print("Save Edges to %s" % save_fname)
    #         self.model.save_pred_to_disk(edge_map[0], save_fname)

    #         edge_maps.append(edge_map)

    #     edge_maps = np.concatenate(edge_maps, axis=0)
    #     return torch.from_numpy(edge_maps)

    def apply_expert_one_frame(self, rgb_frame):
        resized_rgb_frame = cv2.resize(np.array(rgb_frame),
                                       (W, H)).astype(np.float32)
        input_tensor = self.trans_totensor(resized_rgb_frame)[None].to(self.device)
        up_edge, up_sal, up_sal_f = self.net_bone(input_tensor)
        pred = np.squeeze(torch.sigmoid(up_sal_f[-1]).cpu().data.numpy())

        # out shape: expert.n_maps x 256 x 256
        return pred[None]
