import os
import sys
import time
from math import exp

import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from skimage import color
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

sys.path.insert(0,
                os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import multiprocessing

EPSILON = 0.00001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
COLORS_SHORT = ('red', 'blue', 'yellow', 'magenta', 'green', 'indigo',
                'darkorange', 'cyan', 'pink', 'yellowgreen', 'chocolate',
                'lightsalmon', 'lime', 'silver', 'gainsboro', 'gold', 'coral',
                'aquamarine', 'lightcyan', 'oldlace', 'darkred', 'snow')


def img_for_plot(img, dst_id, is_gt=False):
    '''
    img shape NCHW, ex: torch.Size([3, 1, 256, 256])
    '''
    img = img.clone()
    n, c, _, _ = img.shape
    if c == 2:
        img = img[:, 0:1]
        c = 1
    if dst_id.find("sem_seg") >= 0:
        tasko_labels = img
        all_classes = 12
        for idx in range(all_classes):
            tasko_labels[:, 0, 0, idx] = idx
            tasko_labels[:, 0, idx, 0] = idx

        result = color.label2rgb((tasko_labels[:, 0]).data.cpu().numpy(),
                                 colors=COLORS_SHORT,
                                 bg_label=0).transpose(0, 3, 1, 2)
        img = torch.from_numpy(result.astype(np.float32)).contiguous()
        c = 3

    # # v1. normalize per channel
    # img_view = img.view(n, c, -1)
    # min_img = img_view.min(axis=2)[0][:, :, None, None]
    # max_img = img_view.max(axis=2)[0][:, :, None, None]
    # return (img - min_img) / (max_img - min_img)

    # normalize per all input (all channels)
    img_view = img.view(n, -1)
    min_img = img_view.min(axis=1)[0][:, None, None, None]
    max_img = img_view.max(axis=1)[0][:, None, None, None]
    return (img - min_img) / (max_img - min_img)


def get_gaussian_filter(n_channels, win_size, sigma):
    # build gaussian filter for SSIM
    h_win_size = win_size // 2
    yy, xx = torch.meshgrid([
        torch.arange(-h_win_size, h_win_size + 1, dtype=torch.float32),
        torch.arange(-h_win_size, h_win_size + 1, dtype=torch.float32)
    ])
    g_filter = torch.exp((-0.5) * ((xx**2 + yy**2) / (2 * sigma**2)))
    g_filter = g_filter.unsqueeze(0).unsqueeze(0)
    g_filter = g_filter.repeat(n_channels, 1, 1, 1)
    g_filter = g_filter / torch.sum(g_filter)
    return g_filter


class DummySummaryWriter:
    def __init__(*args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, *args, **kwargs):
        return self


class SimScore_SSIM():
    def __init__(self, n_channels, win_size, reduction=False):
        super(SimScore_SSIM, self).__init__()
        self.n_channels = n_channels
        self.win_size = win_size
        self.sigma = self.win_size / 7
        self.reduction = reduction
        self.g_filter = get_gaussian_filter(self.n_channels, self.win_size,
                                            self.sigma).to(device)

    def get_similarity_score(self, batch1, batch2):
        mu1 = torch.nn.functional.conv2d(batch1,
                                         self.g_filter,
                                         padding=self.win_size // 2,
                                         groups=self.n_channels)
        mu2 = torch.nn.functional.conv2d(batch2,
                                         self.g_filter,
                                         padding=self.win_size // 2,
                                         groups=self.n_channels)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = torch.nn.functional.conv2d(batch1 * batch1,
                                               self.g_filter,
                                               padding=self.win_size // 2,
                                               groups=self.n_channels) - mu1_sq
        sigma1_sq = torch.abs(sigma1_sq)

        sigma2_sq = torch.nn.functional.conv2d(batch2 * batch2,
                                               self.g_filter,
                                               padding=self.win_size // 2,
                                               groups=self.n_channels) - mu2_sq
        sigma2_sq = torch.abs(sigma2_sq)

        sigma12 = torch.nn.functional.conv2d(batch1 * batch2,
                                             self.g_filter,
                                             padding=self.win_size // 2,
                                             groups=self.n_channels) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) *
                    (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                           (sigma1_sq + sigma2_sq + C2))
        if self.reduction:
            res = ssim_map.view((ssim_map.shape[0], ssim_map.shape[1],
                                 -1)).mean(2).mean(1).sum()
        else:
            res = ssim_map
        res = torch.clamp(res, min=0)
        return res


class SimScore_MSSIM():
    def __init__(self, n_channels, win_sizes, reduction=False):
        super(SimScore_MSSIM, self).__init__()
        self.n_channels = n_channels
        self.win_sizes = win_sizes
        self.sigmas = self.win_sizes / 7
        self.reduction = reduction

        self.g_filters = []
        for i in range(len(self.win_sizes)):
            self.g_filters.append(
                get_gaussian_filter(self.n_channels, self.win_sizes[i],
                                    self.sigmas[i]).to(device))

    def get_similarity_score_aux(self, batch1, batch2, g_filter, win_size):
        mu1 = torch.nn.functional.conv2d(batch1,
                                         g_filter,
                                         padding=win_size // 2,
                                         groups=self.n_channels)
        mu2 = torch.nn.functional.conv2d(batch2,
                                         g_filter,
                                         padding=win_size // 2,
                                         groups=self.n_channels)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = torch.nn.functional.conv2d(batch1 * batch1,
                                               g_filter,
                                               padding=win_size // 2,
                                               groups=self.n_channels) - mu1_sq
        sigma1_sq = torch.abs(sigma1_sq)

        sigma2_sq = torch.nn.functional.conv2d(batch2 * batch2,
                                               g_filter,
                                               padding=win_size // 2,
                                               groups=self.n_channels) - mu2_sq
        sigma2_sq = torch.abs(sigma2_sq)

        sigma12 = torch.nn.functional.conv2d(batch1 * batch2,
                                             g_filter,
                                             padding=win_size // 2,
                                             groups=self.n_channels) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) *
                    (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                           (sigma1_sq + sigma2_sq + C2))
        if self.reduction:
            res = ssim_map.view((ssim_map.shape[0], ssim_map.shape[1],
                                 -1)).mean(2).mean(1).sum()
        else:
            res = ssim_map
        res = torch.clamp(res, min=0)
        return res

    def get_similarity_score(self, batch1, batch2):
        res = self.get_similarity_score_aux(batch1, batch2, self.g_filters[0],
                                            self.win_sizes[0])
        for i in range(len(self.g_filters)):
            res = res + self.get_similarity_score_aux(
                batch1, batch2, self.g_filters[i], self.win_sizes[i])
        res = res / len(self.g_filters)
        return res


class SimScore_L1():
    def __init__(self, reduction=False):
        super(SimScore_L1, self).__init__()
        if not reduction:
            self.reduction = 'none'
        else:
            self.reduction = 'mean'

    def get_similarity_score(self, batch1, batch2):
        res = torch.nn.functional.l1_loss(batch1,
                                          batch2,
                                          reduction=self.reduction)
        return res


class SimScore_Equal():
    def __init__(self, reduction=False):
        super(SimScore_Equal, self).__init__()
        self.reduction = reduction

    def get_similarity_score(self, batch1, batch2):
        bs, n_chs, h, w = batch1.shape
        if self.reduction:
            return torch.ones((n_chs, h, w)).cuda()
        else:
            return torch.ones((bs, n_chs, h, w)).cuda()


class SimScore_PSNR():
    def __init__(self, reduction=False):
        super(SimScore_PSNR, self).__init__()
        if reduction:
            self.reduction = 'mean'
        else:
            self.reduction = 'none'

    def get_similarity_score(self, batch1, batch2):
        mse = torch.nn.functional.mse_loss(batch1,
                                           batch2,
                                           reduction=self.reduction)
        norm_dist = torch.log10(1 / (mse + EPSILON))
        res = 1 - norm_dist / (norm_dist.max() + EPSILON)
        return res


class SimScore_LPIPS():
    def __init__(self, n_channels):
        super(SimScore_LPIPS, self).__init__()
        self.n_channels = n_channels
        self.lpips_net = lpips.LPIPS(net='squeeze', spatial=True).to(device)
        # LPIPS_NETS['lpips_alex'] = lpips.LPIPS(net='alex', spatial=True).to(device)
        # LPIPS_NETS['lpips_squeeze'] = lpips.LPIPS(net='squeeze',
        #                                           spatial=True).to(device)
    def get_similarity_score(self, batch1, batch2):
        distance = self.lpips_net.forward(batch1, batch2)
        distance = distance.repeat(1, self.n_channels, 1, 1)
        distance = 1 - distance
        return distance


class EnsembleFilter_TwdExpert(torch.nn.Module):
    def __init__(self, n_channels, similarity_fct='ssim', threshold=0.5):
        super(EnsembleFilter_TwdExpert, self).__init__()
        self.threshold = threshold
        self.similarity_fct = similarity_fct
        self.n_channels = n_channels

        if similarity_fct == 'ssim':
            self.similarity_model = SimScore_SSIM(self.n_channels, 11)
        elif similarity_fct == 'l1':
            self.similarity_model = SimScore_L1()
        elif similarity_fct == 'equal':
            self.similarity_model = SimScore_Equal()
        elif similarity_fct == 'mssim':
            self.similarity_model = SimScore_MSSIM(self.n_channels,
                                                   np.array([5, 11, 17]))
        elif similarity_fct == 'psnr':
            self.similarity_model = SimScore_PSNR()
        elif similarity_fct == 'lpips':
            self.similarity_model = SimScore_LPIPS(self.n_channels)

    def forward_mean(self, data, weights):
        data = data * weights
        return torch.sum(data, -1)

    def forward_median(self, data, weights):
        bs, n_chs, h, w, n_exps = data.shape

        data = data.view(bs * n_chs * h * w, n_exps)
        weights = weights.view(bs * n_chs * h * w, n_exps)
        indices = torch.argsort(data, 1)

        data = data[torch.arange(bs * n_chs * h * w).unsqueeze(1).repeat(
            (1, n_exps)), indices]
        weights = weights[torch.arange(bs * n_chs * h * w).unsqueeze(1).repeat(
            (1, n_exps)), indices]
        weights = torch.cumsum(weights, 1)

        weights[weights < 0.5] = 2
        _, indices = torch.min(weights, 1, keepdim=True)
        data = data[torch.arange(bs * n_chs * h * w).unsqueeze(1), indices]
        data = data.contiguous().view(bs, n_chs, h, w)
        return data

    def twd_expert_distances(self, data):
        bs, n_chs, h, w, n_tasks = data.shape

        similarity_maps = []
        for i in range(n_tasks):
            similarity_map = self.similarity_model.get_similarity_score(
                data[..., -1], data[..., i])
            similarity_maps.append(similarity_map)

        similarity_maps = torch.stack(similarity_maps, 0)

        return similarity_maps

    def forward(self, data, dst_domain_name):
        similarity_maps = self.twd_expert_distances(data)
        similarity_maps = similarity_maps.permute(1, 2, 3, 4, 0)

        data[similarity_maps < self.threshold] = 0
        similarity_maps[similarity_maps < self.threshold] = 0

        sum_ = torch.sum(similarity_maps, -1)[:, :, :, :, None]
        sum_[sum_ == 0] = 1
        weights = similarity_maps / sum_

        if dst_domain_name == 'edges':
            return self.forward_mean(data, weights)
        else:
            return self.forward_median(data, weights)
