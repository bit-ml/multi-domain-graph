# import sys

# print(sys.path)
from experts.depth_expert import DepthModel
from experts.edges_expert import EdgesModel
from experts.halftone_expert import HalftoneModel
from experts.normals_expert import SurfaceNormalsModel
from experts.rgb_expert import RGBModel
from experts.tracking1_expert import Tracking1Model
from experts.raft_of_experts import RaftTest


class Experts:
    def __init__(self, full_experts=True):
        # self.methods = [DepthModel(), EdgesModel(), HalftoneModel(), Tracking1Model(), SurfaceNormalsModel()]
        # self.methods = [SurfaceNormalsModel(), HalftoneModel()]
        self.methods = [
            RGBModel(full_experts),
            DepthModel(full_experts),
            EdgesModel(full_experts),
            SurfaceNormalsModel(full_experts),
            HalftoneModel(full_experts),
            Tracking1Model(full_experts),
            RaftTest(full_experts)
        ]

    def rgb_inference(self, rgb_frames):
        output_maps = []
        for expert in self.methods:
            output_map = expert.apply_expert(rgb_frames)
            output_maps.append(output_map)
        return output_maps
