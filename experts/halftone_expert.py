# https://github.com/philgyford/python-halftone
import cv2
import numpy as np
import torch
from PIL import Image
from experts.halftone.halftone import Halftone

W, H = 256, 256


class HalftoneModel():
    def __init__(self, full_expert=True, style=0):
        # if full_expert:
        # self.model = Halftone()
        self.style = style

        if style == 0:
            self.n_maps = 1
            self.domain_name = "halftone_gray"
        elif style == 1:
            self.n_maps = 3
            self.domain_name = "halftone_rgb"
        elif style == 2:
            self.n_maps = 4
            self.domain_name = "halftone_cmyk"
        elif style == 3:
            self.n_maps = 1
            self.domain_name = "halftone_rot_gray"
        self.str_id = "basic"
        self.identifier = self.domain_name + "_" + self.str_id

    def apply_expert_batch(self, batch_rgb_frames):
        halftone_maps = []
        for idx, rgb_frame in enumerate(batch_rgb_frames):
            from PIL import Image
            rgb_frame = Image.fromarray(np.uint8(rgb_frame))
            halftone_map = self.apply_expert_one_frame(rgb_frame)
            halftone_maps.append(np.array(halftone_map))
        halftone_maps = np.array(halftone_maps).astype('float32')
        return halftone_maps

    '''
    def apply_expert(self, rgb_frames):
        halftone_maps = []
        for idx, rgb_frame in enumerate(rgb_frames):
            from PIL import Image
            rgb_frame = Image.fromarray(np.uint8(rgb_frame))
            halftone_map = self.apply_expert_one_frame(rgb_frame)
            # h.make(
            #     angles=[15, 75, 0, 45],
            #     antialias=True,
            #     filename_addition='_new',
            #     percentage=50,
            #     sample=5,
            #     scale=2,
            #     style='color'
            # )
            # save_fname = "halftone_test.png"
            # print("Save Halftone to %s" % save_fname)
            # halftone_map.save(save_fname)
            halftone_maps.append(np.array(halftone_map))

        #halftone_maps = np.concatenate(halftone_maps, axis=0)
        #return torch.from_numpy(halftone_maps)
        return halftone_maps

    def apply_expert_one_frame(self, rgb_frame):
        resized_rgb_frame = rgb_frame.resize((W, H))
        halftone_map = Halftone(resized_rgb_frame, self.style).make()

        if self.style == 0 or self.style == 3:
            halftone_map = np.array(halftone_map)[None, :, :] / 255.
        else:
            halftone_map = np.array(halftone_map).transpose(2, 0, 1) / 255.

        return halftone_map
    '''