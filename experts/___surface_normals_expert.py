# https://github.com/nnaisense/deep-iterative-surface-normal-estimation 
# python normals_pcpnetdata_eval.py --model_name='network_k64.pt' --k_test=64 --iterations=4
# We provide models trained on the PCPNet train dataset with k=32,48,64,96,128 in 'trained_models/'.
import cv2
import numpy as np
import torch

from experts.surface_normals.surface_normals import NormalEstimation, test

W, H = 400, 400


class SurfaceNormalsModel():
    def __init__(self):
        self.model = NormalEstimation()
        self.k_size = 64
        self.iters = 2
        model_path = "experts/models/surface_normals_network_k%d.pt" % self.k_size
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def apply_expert(self, rgb_frames):
        halftone_maps = []
        for idx, rgb_frame in enumerate(rgb_frames):
            resized_rgb_frame = rgb_frame.resize((W, H))

            test(self.model, self.iters, self.k_size, pos, batch)

            save_fname = "surface_normals_test.png"
            print("Save Surface Normals to %s" % save_fname)
        #     halftone_map.save(save_fname)
        #     halftone_maps.append(np.array(halftone_map))

        # halftone_maps = np.concatenate(halftone_maps, axis=0)
        return halftone_maps#torch.from_numpy(halftone_maps)
