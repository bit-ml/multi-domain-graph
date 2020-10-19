# use SGDepth code for depth expert - https://github.com/ifnspaml/SGDepth

import torch

from experts.depth.arguments import InferenceEvaluationArguments
from experts.depth.inference import Inference


# class DepthModel(GeneralExpert):
class DepthModel():
    def __init__(self):
        opt = InferenceEvaluationArguments().parse()
        opt.model_path = "experts/models/depth.pth"
        opt.inference_resize_height = 192
        opt.inference_resize_width = 640
        self.model = Inference(opt)
        self.model.eval()

    def apply_expert(self, rgb_frames):
        depth_maps = []
        for idx, rgb_frame in enumerate(rgb_frames):
            depth_map, segm_map = self.model.inference(rgb_frame)

            save_fname = "depth_test.png"
            print("Save depth in %s" % save_fname)
            self.model.save_pred_to_disk(depth_map, segm_map, save_fname)

            depth_maps.append(depth_map)
        return torch.cat(depth_maps, dim=0)
