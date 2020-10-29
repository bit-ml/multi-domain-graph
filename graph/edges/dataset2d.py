import glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Domain2DDataset(Dataset):
    def __init__(self, rgbs_dir_path, experts):
        super(Domain2DDataset, self).__init__()
        self.experts = experts

        # load all rgbs paths
        # self.rgb_paths = glob.glob("%s/*/*111.jpg" % rgbs_dir_path)
        # np.save("rgb_paths.npy", self.rgb_paths)
        self.rgb_paths = np.load("rgb_paths.npy")[:20]
        print(self.rgb_paths[0])

    def __getitem__(self, index):
        rgb_path = self.rgb_paths[index]
        img = Image.open(rgb_path)
        experts_output = []

        for expert in self.experts:
            e_out = expert.apply_expert_one_frame(img)
            experts_output.append(e_out)

        # return torch.from_numpy(np.array(img)), experts_output
        # TODO: normalize?!!
        return experts_output

    def __len__(self):
        return len(self.rgb_paths)
