# https://github.com/philgyford/python-halftone
import cv2
import numpy as np
import torch

from experts.halftone.halftone import Halftone

W, H = 400, 400


class HalftoneModel():
    # def __init__(self):
    #     self.model = Halftone()

    def apply_expert(self, rgb_frames):
        halftone_maps = []
        for idx, rgb_frame in enumerate(rgb_frames):
            resized_rgb_frame = rgb_frame.resize((W, H))
            halftone_map = Halftone(resized_rgb_frame).make()
            # h.make(
            #     angles=[15, 75, 0, 45],
            #     antialias=True,
            #     filename_addition='_new',
            #     percentage=50,
            #     sample=5,
            #     scale=2,
            #     style='color'
            # )

            save_fname = "halftone_test.png"
            print("Save Halftone to %s" % save_fname)
            halftone_map.save(save_fname)
            halftone_maps.append(np.array(halftone_map))

        halftone_maps = np.concatenate(halftone_maps, axis=0)
        return torch.from_numpy(halftone_maps)
