# use SGDepth code for depth expert - https://github.com/ifnspaml/SGDepth
import os
import torch

from experts.depth.arguments import InferenceEvaluationArguments
from experts.depth.inference import Inference

W, H = 256, 256

current_dir_name = os.path.dirname(os.path.realpath(__file__))
depth_model_path = os.path.join(current_dir_name, 'models/depth_sgdepth.pth')


# class DepthModel(GeneralExpert):
class DepthModel():
    def __init__(self, full_expert=True):
        if full_expert:
            opt = InferenceEvaluationArguments().parse()
            opt.model_path = depth_model_path  #"experts/models/depth_sgdepth.pth"
            # the model is trained for this inference size!!
            opt.inference_resize_height = H
            opt.inference_resize_width = W
            self.model = Inference(opt)
            self.model.model.eval()
        self.domain_name = "depth"
        self.n_maps = 1
        self.str_id = "sgdepth"
        self.identifier = "depth_sgdepth"

    def apply_expert(self, rgb_frames):
        depth_maps = []
        for idx, rgb_frame in enumerate(rgb_frames):
            depth_map, segm_map = self.model.inference(rgb_frame)

            #save_fname = "depth_test.png"
            #print("Save depth in %s" % save_fname)
            #self.model.save_pred_to_disk(depth_map, segm_map, save_fname)

            depth_maps.append(depth_map[0].cpu().numpy())
        return depth_maps  #torch.cat(depth_maps, dim=0).cpu().numpy()

    def apply_expert_one_frame(self, rgb_frame):
        depth_map, segm_map = self.model.inference(rgb_frame)
        depth_map = depth_map.data.cpu()[0]
        return depth_map
