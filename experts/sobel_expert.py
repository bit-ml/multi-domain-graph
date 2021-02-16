import torch
from utils.utils import get_gaussian_filter

from experts.basic_expert import BasicExpert


class SobelEdgesExpert(BasicExpert):
    def __init__(self, sigma, full_expert=True):
        self.domain_name = "edges"
        self.str_id = "sobel"
        self.identifier = self.domain_name + "_" + self.str_id
        self.n_maps = 1
        self.sigma = sigma
        self.win_size = 2 * (int(2.0 * sigma + 0.5)) + 1
        self.n_channels = 1

        if full_expert:
            self.sobel_filter = torch.FloatTensor([[1, 2, 1], [0, 0, 0],
                                                   [-1, -2, -1]])[None, None]
            self.g_filter = get_gaussian_filter(n_channels=self.n_channels,
                                                win_size=self.win_size,
                                                sigma=sigma).float()

    def apply_expert_batch(self, batch_rgb_frames):
        batch_rgb_frames = batch_rgb_frames.permute(0, 3, 1, 2) / 255.
        batch_rgb_frames = batch_rgb_frames.mean(axis=1, keepdim=True)

        blurred = torch.nn.functional.conv2d(batch_rgb_frames,
                                             self.g_filter,
                                             padding=self.win_size // 2,
                                             groups=self.n_channels).float()
        sx = torch.nn.functional.conv2d(blurred,
                                        self.sobel_filter,
                                        padding=self.win_size // 2,
                                        groups=self.n_channels)
        sy = torch.nn.functional.conv2d(blurred,
                                        self.sobel_filter.permute(
                                            (0, 1, 3, 2)),
                                        padding=self.win_size // 2,
                                        groups=self.n_channels)
        edges = torch.hypot(sx, sy)

        return edges.data.cpu().numpy()


class SobelEdgesExpertSigmaLarge(SobelEdgesExpert):
    def __init__(self, full_expert=True):
        SobelEdgesExpert.__init__(self, sigma=4., full_expert=full_expert)
        self.str_id = "sobel_large"
        self.identifier = self.domain_name + "_" + self.str_id


class SobelEdgesExpertSigmaMedium(SobelEdgesExpert):
    def __init__(self, full_expert=True):
        SobelEdgesExpert.__init__(self, sigma=1., full_expert=full_expert)
        self.str_id = "sobel_medium"
        self.identifier = self.domain_name + "_" + self.str_id


class SobelEdgesExpertSigmaSmall(SobelEdgesExpert):
    def __init__(self, full_expert=True):
        SobelEdgesExpert.__init__(self, sigma=0.1, full_expert=full_expert)
        self.str_id = "sobel_small"
        self.identifier = self.domain_name + "_" + self.str_id
