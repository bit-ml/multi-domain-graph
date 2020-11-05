class OFModel():
    def __init__(self, full_expert=True):
        self.domain_name = "of"
        self.n_maps = 2
        self.str_id = "of_fwd_raft"

    # def apply_expert(self, rgb_frames):
    #     # todo resize
    #     return np.array(rgb_frames) / 255.

    # def apply_expert_one_frame(self, rgb_frame):
    #     rgb_frame = rgb_frame.resize((W, H))
    #     return np.array(rgb_frame, dtype=np.float32).transpose(2, 0, 1) / 255.
