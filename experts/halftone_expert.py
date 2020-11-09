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

        self.domain_name = "halftone"
        if style==0:
            self.n_maps = 1
            self.str_id = "halftone_gray_basic"
        elif style==1:
            self.n_maps = 3
            self.str_id = "halftone_rgb_basic"
        elif style==2:
            self.n_maps = 4
            self.str_id = "halftone_cmyk_basic"
        elif style==3:
            self.n_maps = 1
            self.str_id = "halftone_rot_gray_basic"
        
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
       
        if self.style==0 or self.style==3:
            halftone_map = np.array(halftone_map)[None,:,:]/255.
        else:
            halftone_map = np.array(halftone_map).transpose(2, 0, 1)/255.

        return halftone_map
