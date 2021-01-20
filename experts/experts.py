# import sys

# print(sys.path)
from experts.depth_expert import DepthModel, DepthModelXTC
from experts.edges_expert import EdgesModel
from experts.halftone_expert import HalftoneModel
from experts.normals_expert import SurfaceNormalsXTC
from experts.raft_of_expert import RaftModel
from experts.rgb_expert import RGBModel
from experts.saliency_seg_expert import SaliencySegmModel
from experts.tracking1_expert import Tracking1Model


class Experts:
    def __init__(self, full_experts=True):
        # self.methods = [DepthModel(), EdgesModel(), HalftoneModel(), Tracking1Model(), SurfaceNormalsXTC()]
        # self.methods = [SurfaceNormalsXTC(), HalftoneModel()]
        self.methods = [
            RGBModel(full_experts),
            DepthModelXTC(full_experts),
            SurfaceNormalsXTC(full_experts),
            EdgesModel(full_experts),
            SaliencySegmModel(full_experts),
            HalftoneModel(full_experts, 0),

            # Tracking1Model(full_experts),
            # RaftModel(full_experts, 1),
            # DepthModel(full_experts),
        ]

    '''
    def rgb_inference(self, rgb_frames):
        output_maps = []
        for expert in self.methods:
            output_map = expert.apply_expert(rgb_frames)
            output_maps.append(output_map)
        return output_maps
    '''
