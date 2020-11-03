import numpy as np

W, H = 256, 256


class RGBModel():
    def __init__(self, full_expert=True):
        self.domain_name = "rgb"
        self.n_maps = 3
        self.str_id = "rgb"

    def apply_expert(self, rgb_frames):
        # todo resize
        return np.array(rgb_frames) / 255.

    def apply_expert_one_frame(self, rgb_frame):
        rgb_frame = rgb_frame.resize((W, H))
        return np.array(rgb_frame, dtype=np.float32).transpose(2, 0, 1) / 255.
