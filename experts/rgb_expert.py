import numpy as np

from experts.basic_expert import BasicExpert

W, H = 256, 256


class RGBModel(BasicExpert):
    def __init__(self, full_expert=True):
        self.domain_name = "rgb"
        self.n_maps = 3
        self.str_id = ""
        self.identifier = "rgb"


    def apply_expert_batch(self, batch_rgb_frames):
        batch_rgb_frames = batch_rgb_frames.numpy()
        batch_rgb_frames = batch_rgb_frames.astype('float32')
        batch_rgb_frames = batch_rgb_frames / 255.0
        batch_rgb_frames = np.moveaxis(batch_rgb_frames, 3, 1)
        return batch_rgb_frames

    '''
    def apply_expert(self, rgb_frames):
        # todo resize
        return np.array(rgb_frames) / 255.

    def apply_expert_one_frame(self, rgb_frame):
        rgb_frame = rgb_frame.resize((W, H))
        return np.array(rgb_frame, dtype=np.float32).transpose(2, 0, 1) / 255.
    '''
