# import sys

# print(sys.path)
import numpy as np

from experts.depth_expert import DepthModel, DepthModelXTC
from experts.edges_expert import EdgesModel
from experts.halftone_expert import HalftoneModel
from experts.normals_expert import SurfaceNormalsXTC
from experts.raft_of_expert import RaftModel
from experts.rgb_expert import RGBModel
from experts.saliency_seg_expert import SaliencySegmModel
from experts.tracking1_expert import Tracking1Model


class Experts:
    def __init__(self, full_experts=True, selector_map=None):

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
        if selector_map is None:
            selector_map = np.arange(len(self.methods))

        self.methods = np.array(self.methods)[selector_map].tolist()

        print("==================")
        print("USED", len(self.methods), "EXPERTS:",
              [method.__class__.__name__ for method in self.methods])
        print("==================")
