# use SGDepth code for depth expert - https://github.com/xavysp/DexiNed/blob/master/DexiNed-Pytorch/

import torch
import cv2
import tensorflow as tf 
import numpy as np
from experts.edges.model import DexiNed


W, H = 400, 400
class EdgesModel():
    def __init__(self):
        checkpoint_path = "experts/models/edges_dexined23.h5"
        device = "gpu" if torch.cuda.is_available() else "cpu"
        rgbn_mean = np.array([103.939, 116.779, 123.68, 137.86])[None, None, None, :]
        input_shape = (1, H, W, 3)
        self.model = DexiNed(rgb_mean=rgbn_mean)
        self.model.build(input_shape=input_shape)
        self.model.load_weights(checkpoint_path)

    def apply_expert(self, rgb_frames):
        edge_maps = []
        for idx, rgb_frame in enumerate(rgb_frames):
            resized_rgb_frame = cv2.resize(np.array(rgb_frame), (W, H)).astype(np.float32)
            preds = self.model(resized_rgb_frame, training=False)
            edge_map = tf.sigmoid(preds).numpy()[:, :, :, 0]

            save_fname = "edge_test.png"
            print("Save Edges to %s" % save_fname)
            self.model.save_pred_to_disk(edge_map[0], save_fname)

            edge_maps.append(edge_map)

        edge_maps = np.concatenate(edge_maps, axis=0)
        return torch.from_numpy(edge_maps)
