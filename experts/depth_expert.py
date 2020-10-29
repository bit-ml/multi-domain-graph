# use SGDepth code for depth expert - https://github.com/ifnspaml/SGDepth

import torch

from experts.depth.arguments import InferenceEvaluationArguments
from experts.depth.inference import Inference

W, H = 256, 256


# class DepthModel(GeneralExpert):
class DepthModel():
    def __init__(self):
        opt = InferenceEvaluationArguments().parse()
        opt.model_path = "experts/models/depth_sgdepth.pth"
        # the model is trained for this inference size!!
        opt.inference_resize_height = H
        opt.inference_resize_width = W
        self.model = Inference(opt)
        self.model.model.eval()
        self.domain_name = "depth"
        self.n_maps = 1
        self.str_id = "sgdepth"

    def apply_expert(self, rgb_frames):
        depth_maps = []
        for idx, rgb_frame in enumerate(rgb_frames):
            depth_map, segm_map = self.model.inference(rgb_frame)

            save_fname = "depth_test.png"
            print("Save depth in %s" % save_fname)
            self.model.save_pred_to_disk(depth_map, segm_map, save_fname)

            depth_maps.append(depth_map)
        return torch.cat(depth_maps, dim=0)
