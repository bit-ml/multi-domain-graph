# White-Box Cartoonization https://github.com/SystemErrorWang/White-box-Cartoonization
import os
import cv2

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from experts.basic_expert import BasicExpert
from experts.cartoon import network
from experts.cartoon import guided_filter

W, H = 256, 256

current_dir_name = os.path.dirname(os.path.realpath(__file__))
cartoon_model_path = os.path.join(current_dir_name, 'models/cartoon_wb')


class CartoonWB(BasicExpert):
    def __init__(self, full_expert=True):
        #if full_expert:
        self.domain_name = "cartoon"
        self.n_maps = 3
        self.str_id = "wb"
        self.identifier = self.domain_name + "_" + self.str_id

    def apply_expert_batch(self, batch_rgb_frames):
        self.input_photo = tf.placeholder(tf.float32, [None, None, None, 3])
        self.network_out = network.unet_generator(self.input_photo)
        self.final_out = guided_filter.guided_filter(self.input_photo,
                                                     self.network_out,
                                                     r=1,
                                                     eps=5e-3)

        all_vars = tf.trainable_variables()
        gene_vars = [var for var in all_vars if 'generator' in var.name]
        self.saver = tf.train.Saver(var_list=gene_vars)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess,
                           tf.train.latest_checkpoint(cartoon_model_path))

        batch_rgb_frames = batch_rgb_frames.numpy()
        batch_rgb_frames = batch_rgb_frames.astype(np.float32) / 127.5 - 1
        output = self.sess.run(self.final_out,
                               feed_dict={self.input_photo: batch_rgb_frames})
        output = (np.squeeze(output) + 1) * 127.5
        output = np.clip(output, 0, 255).astype(np.float32)
        output = output / 255.

        return output
