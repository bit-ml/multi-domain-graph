import glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Domain2DDataset(Dataset):
    def __init__(self, rgbs_path, experts_path, dataset_path, experts):
        super(Domain2DDataset, self).__init__()
        self.experts = experts

        # load all rgbs paths
        self.rgb_paths = sorted(
            glob.glob("%s/%s*/*333.jpg" % (rgbs_path, dataset_path)))

        self.e1_output_path = sorted(
            glob.glob("%s/%s/%s/*/*333.npy" %
                      (experts_path, self.experts[0].str_id, dataset_path)))
        self.e2_output_path = sorted(
            glob.glob("%s/%s/%s/*/*333.npy" %
                      (experts_path, self.experts[1].str_id, dataset_path)))

        print("Dataset size", len(self.rgb_paths))
        assert (len(self.rgb_paths) == len(self.e1_output_path) == len(
            self.e2_output_path))
        # print(self.rgb_paths[0])
        # print(self.e1_output_path[0])
        # print(self.e2_output_path[0])

    def __getitem__(self, index):
        # rgb_path = self.rgb_paths[index]
        # img = Image.open(rgb_path)
        # experts_output = []
        # for expert in self.experts:
        #     e_out = expert.apply_expert_one_frame(img)
        #     experts_output.append(e_out)

        oe1 = np.load(self.e1_output_path[index])
        oe2 = np.load(self.e2_output_path[index])

        # return torch.from_numpy(np.array(img)), experts_output
        # TODO: normalize?!!
        return oe1, oe2

    def __len__(self):
        return len(self.rgb_paths)
