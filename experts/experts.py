# import sys

# print(sys.path)
from experts.depth_expert import DepthModel
from experts.edges_expert import EdgesModel
from experts.halftone_expert import HalftoneModel


class Experts:
    def __init__(self):
        # self.experts = [DepthModel(), EdgesModel(), HalftoneModel()]
        self.experts = [HalftoneModel()]

    def rgb_inference(self, rgb_frames):
        output_maps = []
        for expert in self.experts:
            output_map = expert.apply_expert(rgb_frames)
            output_maps.append(output_map)
        return output_maps
