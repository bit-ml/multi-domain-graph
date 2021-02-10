# import sys

# print(sys.path)
import numpy as np

from experts.depth_expert import DepthModelXTC
from experts.edges_expert import EdgesModel
from experts.grayscale_expert import Grayscale
from experts.halftone_expert import HalftoneModel
from experts.hsv_expert import HSVExpert
from experts.normals_expert import SurfaceNormalsXTC
from experts.rgb_expert import RGBModel
from experts.semantic_segmentation_expert import SSegHRNet


class Experts:
    def __init__(self, dataset_name, full_experts=True, selector_map=None):
        self.methods = [
            RGBModel(full_experts),
            DepthModelXTC(full_experts),
            SurfaceNormalsXTC(dataset_name=dataset_name,
                              full_expert=full_experts),
            EdgesModel(full_experts),
            HalftoneModel(full_experts, 0),
            SSegHRNet(full_experts),
            Grayscale(full_experts),
            HSVExpert(full_experts)
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
