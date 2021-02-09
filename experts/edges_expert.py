# use SGDepth code for depth expert - https://github.com/xavysp/DexiNed/blob/master/DexiNed-Pytorch/
import os

import cv2
import numpy as np
import tensorflow as tf
import torch

from experts.basic_expert import BasicExpert
from experts.edges.model import DexiNed

W, H = 256, 256

current_dir_name = os.path.dirname(os.path.realpath(__file__))
edges_model_path = os.path.join(current_dir_name, 'models/edges_dexined.h5')


class EdgesModel(BasicExpert):
    def __init__(self, full_expert=True):
        if full_expert:
            checkpoint_path = edges_model_path  #"experts/models/edges_dexined23.h5"
            device = "gpu" if torch.cuda.is_available() else "cpu"
            self.device = device
            rgbn_mean = np.array([103.939, 116.779, 123.68,
                                  137.86])[None, None, None, :]
            input_shape = (1, H, W, 3)
            self.model = DexiNed(rgb_mean=rgbn_mean)
            self.model.build(input_shape=input_shape)
            self.model.load_weights(checkpoint_path)

        self.domain_name = "edges"
        self.n_maps = 1
        self.str_id = "dexined"
        self.identifier = self.domain_name + "_" + self.str_id

    def apply_expert_batch(self, batch_rgb_frames):
        edge_maps = []
        batch_rgb_frames = batch_rgb_frames.numpy().astype(np.float32)
        preds = self.model(batch_rgb_frames, training=False)
        edge_maps = tf.sigmoid(preds).numpy()[:, :, :, 0]
        edge_maps = edge_maps[:, None, :, :]
        edge_maps = edge_maps.astype('float32')
        return edge_maps

    '''
    def apply_expert(self, rgb_frames):
        edge_maps = []
        for idx, rgb_frame in enumerate(rgb_frames):
            resized_rgb_frame = cv2.resize(np.array(rgb_frame),
                                           (W, H)).astype(np.float32)
            preds = self.model(resized_rgb_frame, training=False)
            edge_map = tf.sigmoid(preds).numpy()[:, :, :, 0]

            #save_fname = "edge_test.png"
            #print("Save Edges to %s" % save_fname)
            #self.model.save_pred_to_disk(edge_map[0], save_fname)

            edge_maps.append(edge_map)

        #edge_maps = np.concatenate(edge_maps, axis=0)
        #return torch.from_numpy(edge_maps)
        return edge_maps

    def apply_expert_one_frame(self, rgb_frame):
        resized_rgb_frame = cv2.resize(np.array(rgb_frame),
                                       (W, H)).astype(np.float32)
        preds = self.model(resized_rgb_frame, training=False)
        edge_map = tf.sigmoid(preds).numpy()[:, :, :, 0]

        return edge_map
    '''
