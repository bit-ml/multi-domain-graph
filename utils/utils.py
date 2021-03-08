import os
import sys
from math import exp

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from skimage import color
from torch import nn

sys.path.insert(0,
                os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

EPSILON = 0.00001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
COLORS_SHORT = ('red', 'blue', 'yellow', 'magenta', 'green', 'indigo',
                'darkorange', 'cyan', 'pink', 'yellowgreen', 'chocolate',
                'lightsalmon', 'lime', 'silver', 'gainsboro', 'gold', 'coral',
                'aquamarine', 'lightcyan', 'oldlace', 'darkred', 'snow')


# for normals - call with val = 0.788, tol=1e-3, kernel=1
# for depth - call with val=1.0, tol=1e-3, kernel=1
def build_mask(target, val=0.0, tol=1e-3, kernel=1):
    padding = (kernel - 1) // 2
    if target.shape[1] == 1:
        mask = ((target >= val - tol) & (target <= val + tol))
        mask = F.conv2d(mask.float(),
                        torch.ones(1, 1, kernel, kernel, device=mask.device),
                        padding=padding) != 0
        return (~mask).expand_as(target)

    mask1 = (target[:, 0, :, :] >= val - tol) & (target[:, 0, :, :] <=
                                                 val + tol)
    mask2 = (target[:, 1, :, :] >= val - tol) & (target[:, 1, :, :] <=
                                                 val + tol)
    mask3 = (target[:, 2, :, :] >= val - tol) & (target[:, 2, :, :] <=
                                                 val + tol)
    mask = (mask1 & mask2 & mask3).unsqueeze(1)
    mask = F.conv2d(mask.float(),
                    torch.ones(1, 1, kernel, kernel, device=mask.device),
                    padding=padding) != 0
    return (~mask).expand_as(target)


def img_for_plot(img, dst_id):
    '''
    img shape NCHW, ex: torch.Size([3, 1, 256, 256])
    '''
    img = img.clone()
    n, c, _, _ = img.shape
    if dst_id.find("halftone") >= 0:
        if img.shape[1] > 1:
            tasko_labels = img.argmax(dim=1, keepdim=True)
        else:
            tasko_labels = img

        img = tasko_labels
        c = 1
    elif dst_id.find("sem_seg") >= 0:
        if img.shape[1] > 1:
            tasko_labels = img.argmax(dim=1, keepdim=True)
        else:
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
    # img_view = img.view(n, -1)
    # min_img = img_view.min(axis=1)[0][:, None, None, None]
    # max_img = img_view.max(axis=1)[0][:, None, None, None]
    return img  #.clamp(0, 1)  #(img - min_img) / (max_img - min_img)


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


class SSIMLoss(torch.nn.Module):
    def __init__(self, n_channels, win_size, reduction='mean'):
        super(SSIMLoss, self).__init__()
        self.n_channels = n_channels
        self.win_size = win_size
        self.sigma = self.win_size / 7
        self.reduction = reduction
        self.g_filter = get_gaussian_filter(self.n_channels, self.win_size,
                                            self.sigma).to(device)

    def forward(self, batch1, batch2):

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
        if self.reduction == 'mean':
            res = ssim_map.view((ssim_map.shape[0], ssim_map.shape[1],
                                 -1)).mean(2).mean(1).mean()
        res = 1 - ((res + 1) / 2)
        return res


class DummySummaryWriter:
    def __init__(*args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, *args, **kwargs):
        return self


class SimScore_SSIM(nn.Module):
    def __init__(self, n_channels, win_size, reduction=False):
        super(SimScore_SSIM, self).__init__()
        self.n_channels = n_channels
        self.win_size = win_size
        self.sigma = self.win_size / 7
        self.reduction = reduction
        self.g_filter = nn.Parameter(get_gaussian_filter(
            self.n_channels, self.win_size, self.sigma),
                                     requires_grad=False)
        self.conv_filter = torch.nn.Conv2d(in_channels=n_channels,
                                           out_channels=n_channels,
                                           kernel_size=win_size,
                                           padding=win_size // 2,
                                           padding_mode='replicate',
                                           groups=n_channels,
                                           bias=False)
        self.conv_filter.weight = self.g_filter

    def forward(self, batch1, batch2):
        mu1 = self.conv_filter(batch1)
        mu2 = self.conv_filter(batch2)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = self.conv_filter(batch1 * batch1) - mu1_sq
        sigma1_sq = torch.abs(sigma1_sq)
        sigma2_sq = self.conv_filter(batch2 * batch2) - mu2_sq
        sigma2_sq = torch.abs(sigma2_sq)
        sigma12 = self.conv_filter(batch1 * batch2) - mu1_mu2
        C1 = 0.01**2
        C2 = 0.03**2
        ssim_map = ((2 * mu1_mu2 + C1) *
                    (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                           (sigma1_sq + sigma2_sq + C2))
        if self.reduction:
            sim_score = ssim_map.view((ssim_map.shape[0], ssim_map.shape[1],
                                       -1)).mean(2).mean(1).sum()
        else:
            sim_score = ssim_map

        # there seem to be small numerical issues
        sim_score = torch.clamp(sim_score, -1, 1)

        sim_score = (sim_score + 1) / 2
        return 1 - sim_score


class SimScore_MSSIM(torch.nn.Module):
    def __init__(self, n_channels, win_sizes, reduction=False):
        super(SimScore_MSSIM, self).__init__()
        self.n_channels = n_channels
        self.win_sizes = win_sizes
        self.sigmas = self.win_sizes / 7
        self.reduction = reduction

        for i in range(len(self.win_sizes)):
            g_filter_ = get_gaussian_filter(self.n_channels, self.win_sizes[i],
                                            self.sigmas[i])
            self.register_buffer('g_filter_' + str(i), g_filter_)

    def get_similarity_score_aux(self, batch1, batch2, g_filter, win_size):
        mu1 = torch.nn.functional.conv2d(batch1,
                                         g_filter,
                                         padding=win_size // 2,
                                         padding_mode='replicate',
                                         groups=self.n_channels)
        mu2 = torch.nn.functional.conv2d(batch2,
                                         g_filter,
                                         padding=win_size // 2,
                                         padding_mode='replicate',
                                         groups=self.n_channels)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = torch.nn.functional.conv2d(batch1 * batch1,
                                               g_filter,
                                               padding=win_size // 2,
                                               padding_mode='replicate',
                                               groups=self.n_channels) - mu1_sq
        sigma1_sq = torch.abs(sigma1_sq)

        sigma2_sq = torch.nn.functional.conv2d(batch2 * batch2,
                                               g_filter,
                                               padding=win_size // 2,
                                               padding_mode='replicate',
                                               groups=self.n_channels) - mu2_sq
        sigma2_sq = torch.abs(sigma2_sq)

        sigma12 = torch.nn.functional.conv2d(batch1 * batch2,
                                             g_filter,
                                             padding=win_size // 2,
                                             padding_mode='replicate',
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
        # res has now values in range [-1,1]
        res = torch.clamp(res, -1, 1)

        # => values in range [0,1], with 0 worst, 1 best
        res = (res + 1) / 2

        # => [0,1] with 0 best, 1 worst => distance metric
        return 1 - res

    def forward(self, batch1, batch2):
        sim_score = torch.zeros(batch1.shape).cuda()
        for i in range(len(self.win_sizes)):
            sim_score = sim_score + self.get_similarity_score_aux(
                batch1, batch2, self.__getattr__('g_filter_' + str(i)),
                self.win_sizes[i])
        sim_score = sim_score / len(self.win_sizes)

        # sim_score -= sim_score.amin(axis=(2, 3), keepdim=True)
        # sim_score /= (sim_score.amax(axis=(2, 3), keepdim=True) + EPSILON)
        sim_score = (sim_score + 1) / 2

        return 1 - sim_score


class SimScore_L2(nn.Module):
    def __init__(self, reduction=False):
        super(SimScore_L2, self).__init__()
        if not reduction:
            self.reduction = 'none'
        else:
            self.reduction = 'mean'

    def forward(self, batch1, batch2):
        distance = torch.nn.functional.mse_loss(batch1,
                                                batch2,
                                                reduction=self.reduction)

        return distance


class SimScore_L1(nn.Module):
    def __init__(self, reduction=False):
        super(SimScore_L1, self).__init__()
        if reduction:
            self.l1 = torch.nn.L1Loss(reduction='mean')
        else:
            self.l1 = torch.nn.L1Loss(reduction='none')

    def forward(self, batch1, batch2):
        res = self.l1(batch1, batch2)
        # => res >= 0, with 0 best
        return res


class SimScore_L2(nn.Module):
    def __init__(self, reduction=False):
        super(SimScore_L2, self).__init__()
        if reduction:
            self.l2 = torch.nn.MSELoss(reduction='mean')
        else:
            self.l2 = torch.nn.MSELoss(reduction='none')

    def forward(self, batch1, batch2):
        res = self.l2(batch1, batch2)
        # => res >= 0, with 0 best
        return res


class SimScore_Equal(nn.Module):
    def __init__(self):
        super(SimScore_Equal, self).__init__()
        self.ones = torch.ones((70, 3, 256, 256)).cuda()

    def forward(self, batch1, batch2):
        bs, n_chs, h, w = batch1.shape
        bso, n_chso, ho, wo = self.ones.shape
        if bso == bs and n_chso == n_chs and ho == h and wo == w:
            return self.ones

        del self.ones
        self.ones = torch.ones_like(batch1)
        return self.ones


class SimScore_PSNR(nn.Module):
    def __init__(self, reduction=False):
        super(SimScore_PSNR, self).__init__()
        if reduction:
            self.reduction = 'mean'
        else:
            self.reduction = 'none'

    def forward(self, batch1, batch2):
        mse = torch.nn.functional.mse_loss(batch1,
                                           batch2,
                                           reduction=self.reduction)
        norm_dist = torch.log10(1 / (mse + EPSILON))
        return norm_dist


class SimScore_LPIPS(nn.Module):
    def __init__(self, n_channels):
        super(SimScore_LPIPS, self).__init__()
        self.n_channels = n_channels
        self.lpips_net = lpips.LPIPS(net='squeeze',
                                     spatial=True,
                                     verbose=False)
        self.lpips_net.eval()
        self.lpips_net.requires_grad_(False)
        # LPIPS_NETS['lpips_alex'] = lpips.LPIPS(net='alex', spatial=True)
        # LPIPS_NETS['lpips_squeeze'] = lpips.LPIPS(net='squeeze',
        #                                           spatial=True)
    def forward(self, batch1, batch2):
        distance = self.lpips_net.forward(batch1, batch2)
        return distance.repeat(1, self.n_channels, 1, 1)


class EnsembleFilter_TwdExpert(torch.nn.Module):
    def __init__(self,
                 n_channels,
                 dst_domain_name,
                 postprocess_eval,
                 similarity_fcts=['ssim'],
                 kernel_fct='gauss',
                 comb_type='mean',
                 thresholds=[0.5]):
        super(EnsembleFilter_TwdExpert, self).__init__()
        self.thresholds = thresholds
        self.similarity_fcts = similarity_fcts
        self.n_channels = n_channels
        self.dst_domain_name = dst_domain_name
        self.postprocess_eval = postprocess_eval

        self.fct_before_dist_metric = self.scale_maps_before_comparison
        #self.fct_before_dist_metric = lambda x: x

        if kernel_fct == 'flat':
            self.kernel = self.kernel_flat
        elif kernel_fct == 'flat_weighted':
            self.kernel = self.kernel_flat_weighted
        elif kernel_fct == 'gauss':
            self.kernel = self.kernel_gauss

        if comb_type == 'mean':
            self.ens_aggregation_fcn = self.forward_mean
        else:
            self.ens_aggregation_fcn = self.forward_median
        sim_models = []
        for sim_fct in similarity_fcts:
            if sim_fct == 'ssim':
                sim_model = SimScore_SSIM(n_channels, 11)
            elif sim_fct == 'l1':
                sim_model = SimScore_L1()
            elif sim_fct == 'l2':
                sim_model = SimScore_L2()
            elif sim_fct == 'equal':
                sim_model = SimScore_Equal()
            elif sim_fct == 'mssim':
                sim_model = SimScore_MSSIM(n_channels, np.array([5, 11, 17]))
            elif sim_fct == 'psnr':
                sim_model = SimScore_PSNR()
            elif sim_fct == 'lpips':
                sim_model = SimScore_LPIPS(n_channels)

            sim_models.append(sim_model)
        self.similarity_models = torch.nn.ModuleList(sim_models)

    def scale_maps_before_comparison(self, batch1, batch2):
        b1_max = torch.amax(batch1, (1, 2, 3))
        b2_max = torch.amax(batch2, (1, 2, 3))
        b1_min = torch.amin(batch1, (1, 2, 3))
        b2_min = torch.amin(batch2, (1, 2, 3))
        all_max = torch.max(torch.cat((b1_max[None], b2_max[None]), 0),
                            0)[0][:, None, None, None]
        all_min = torch.min(torch.cat((b1_min[None], b2_min[None]), 0),
                            0)[0][:, None, None, None]
        return (batch1 - all_min) / (all_max - all_min), (batch2 - all_min) / (
            all_max - all_min)

    def scale_similarity_maps(self, similarity_maps):
        max_val = torch.amax(similarity_maps, (1, 2, 3, 4))[:, None, None,
                                                            None, None]
        min_val = torch.amin(similarity_maps, (1, 2, 3, 4))[:, None, None,
                                                            None, None]
        return (similarity_maps - min_val) / (max_val - min_val)

    def forward_mean(self, data, weights):
        data = data * weights.cuda()
        return torch.sum(data, -1)

    def forward_median(self, data, weights):
        bs, n_chs, h, w, n_exps = data.shape
        fwd_result = torch.zeros_like(data[..., 0])
        for chan in range(n_chs):
            data_chan = data[:, chan].contiguous()
            data_chan = data_chan.view(bs * h * w, n_exps)
            weights_chan = weights[:, chan].contiguous()
            weights_chan = weights_chan.view(bs * h * w, n_exps)
            indices = torch.argsort(data_chan, 1)

            data_chan = data_chan[torch.arange(bs * h * w).unsqueeze(1).repeat(
                (1, n_exps)), indices]
            weights_chan = weights_chan[torch.arange(bs * h *
                                                     w).unsqueeze(1).repeat(
                                                         (1, n_exps)), indices]
            weights_chan = torch.cumsum(weights_chan, 1)

            weights_chan[weights_chan < 0.5] = 2
            _, indices = torch.min(weights_chan, dim=1, keepdim=True)
            data_chan = data_chan[torch.arange(bs * h * w).unsqueeze(1),
                                  indices]
            data_chan = data_chan.view(bs, h, w)
            fwd_result[:, chan] = data_chan

        return fwd_result

    def kernel_flat(self, chan_dist_maps, meanshift_iter):
        chan_mask = chan_dist_maps > self.thresholds[
            meanshift_iter]  # indicates what we want to remove
        chan_dist_maps[chan_mask] = 0
        chan_dist_maps[~chan_mask] = 1
        return chan_dist_maps

    def kernel_flat_weighted(self, chan_dist_maps, meanshift_iter):
        chan_mask = chan_dist_maps > self.thresholds[
            meanshift_iter]  # indicates what we want to remove
        chan_dist_maps[chan_mask] = 0
        chan_dist_maps[~chan_mask] = 1 - chan_dist_maps[~chan_mask]
        return chan_dist_maps

    def kernel_gauss(self, chan_dist_maps, meanshift_iter):
        chan_dist_maps = torch.exp(-((chan_dist_maps**2) /
                                     (2 * self.thresholds[meanshift_iter]**2)))
        return chan_dist_maps

    def twd_expert_distances(self, data, similarity_model):
        bs, n_chs, h, w, n_tasks = data.shape
        distance_maps = []

        for i in range(n_tasks - 1):
            distance_map = similarity_model(data[..., -1], data[..., i])
            distance_maps.append(distance_map)

        # add expert vs expert
        distance_maps.append(torch.zeros_like(distance_map))

        distance_maps = torch.stack(distance_maps, 0).permute(1, 2, 3, 4, 0)
        return distance_maps

    def scale_distance_maps(self, distance_maps):
        max_val = torch.amax(distance_maps, axis=(1, 2, 3, 4), keepdim=True)
        min_val = torch.amin(distance_maps, axis=(1, 2, 3, 4), keepdim=True)
        return (distance_maps - min_val) / (max_val - min_val + EPSILON)

    def kernel_flat(self, chan_sim_maps, meanshift_iter):
        # indicates what we want to remove
        chan_mask = chan_sim_maps > self.thresholds[meanshift_iter]
        chan_sim_maps[chan_mask] = 0
        chan_sim_maps[~chan_mask] = 1
        return chan_sim_maps

    def kernel_flat_weighted(self, chan_sim_maps, meanshift_iter):
        # indicates what we want to remove
        chan_mask = chan_sim_maps > self.thresholds[meanshift_iter]
        chan_sim_maps = 1 - chan_sim_maps
        chan_sim_maps[chan_mask] = 0
        return chan_sim_maps

    def kernel_gauss(self, chan_sim_maps, meanshift_iter):
        chan_sim_maps = torch.exp(-((chan_sim_maps**2) /
                                    (2 * self.thresholds[meanshift_iter]**2)))
        return chan_sim_maps

    def forward(self, data):
        batch_std = data.std(axis=-1).mean(axis=(1, 2, 3))
        for meanshift_iter in range(len(self.thresholds)):
            bs, n_chs, h, w, n_tasks = data.shape
            distance_maps = self.twd_expert_distances(
                data, self.similarity_models[0])
            distance_maps = self.scale_distance_maps(distance_maps)

            for sim_idx in np.arange(1, len(self.similarity_models)):
                sim_model = self.similarity_models[sim_idx]
                sim_maps = self.twd_expert_distances(data, sim_model)
                sim_maps = self.scale_similarity_maps(sim_maps)
                distance_maps += sim_maps
            distance_maps = self.scale_similarity_maps(distance_maps)

            # kernel: transform distances to similarities
            for chan in range(n_chs):
                distance_maps[:, chan] = self.kernel(distance_maps[:, chan],
                                                     meanshift_iter)
            similarities_maps = distance_maps

            # sum = 1
            sum_ = torch.sum(similarities_maps, dim=-1, keepdim=True)
            sum_[sum_ == 0] = 1
            similarities_maps = similarities_maps / sum_

            ensemble_result = self.ens_aggregation_fcn(data, similarities_maps)

            # 2. clamp/other the ensemble
            # ensemble_result = self.postprocess_eval(ensemble_result)

            data[..., -1] = ensemble_result
        return ensemble_result


class VarianceScore(nn.Module):
    def __init__(self, reduction=False):
        super(VarianceScore, self).__init__()

    def forward(self, batch):
        avg_b = torch.mean(batch, 0)[None]
        avg_b = (batch - avg_b)**2
        avg_b = torch.mean(avg_b, 0)
        return avg_b
