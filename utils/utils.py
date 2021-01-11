import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from math import exp
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0,
                os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import utils.pytorch_ssim


def get_gaussian_filter(n_channels, win_size=11, sigma=1.5):
    h_win_size = win_size // 2
    yy, xx = torch.meshgrid([
        torch.arange(-h_win_size, h_win_size + 1, dtype=torch.float32),
        torch.arange(-h_win_size, h_win_size + 1, dtype=torch.float32)
    ])
    g_filter = torch.exp((-0.5) * ((xx**2 + yy**2) / (2 * sigma**2)))
    g_filter = g_filter.unsqueeze(0).unsqueeze(0)
    g_filter = g_filter.repeat(n_channels, 1, 1, 1)
    return g_filter


def get_ssim_score(batch1, batch2, g_filter, n_channels, win_size):
    mu1 = torch.nn.functional.conv2d(batch1,
                                     g_filter,
                                     padding=win_size // 2,
                                     groups=n_channels)
    mu2 = torch.nn.functional.conv2d(batch2,
                                     g_filter,
                                     padding=win_size // 2,
                                     groups=n_channels)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(
        batch1 * batch1, g_filter, padding=win_size // 2,
        groups=n_channels) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(
        batch2 * batch2, g_filter, padding=win_size // 2,
        groups=n_channels) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(
        batch1 * batch2, g_filter, padding=win_size // 2,
        groups=n_channels) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    res = ssim_map.view((ssim_map.shape[0], ssim_map.shape[1],
                         -1)).mean(2).mean(1).sum()  #.mean()
    return res


def get_correlation_score_ssim(batch_results, correlations, drop_version):
    batch_results = torch.stack(batch_results)
    min_v = torch.min(batch_results)
    max_v = torch.max(batch_results)
    batch_results = (batch_results - min_v) / (max_v - min_v)

    n_tasks = batch_results.shape[0]

    if drop_version == 10 or drop_version == 12 or drop_version == 14 or drop_version == 16:
        win_size = 255
    elif drop_version == 11 or drop_version == 13 or drop_version == 15 or drop_version == 17:
        win_size = 11

    sigma = win_size / 7
    g_filter = get_gaussian_filter(n_channels=batch_results.shape[2],
                                   win_size=win_size,
                                   sigma=sigma).cuda()

    for i in range(n_tasks):
        result = get_ssim_score(batch_results[i], batch_results[i], g_filter,
                                batch_results.shape[2], win_size)
        correlations[i, i] = correlations[i, i] + result
        for j in np.arange(i + 1, n_tasks):
            result = get_ssim_score(batch_results[i], batch_results[j],
                                    g_filter, batch_results.shape[2], win_size)
            correlations[i, j] = correlations[i, j] + result
            correlations[j, i] = correlations[j, i] + result

    return correlations


def get_correlation_score_cosine(batch_results, correlations, drop_version):
    batch_results = torch.stack(batch_results)
    n_tasks = batch_results.shape[0]
    batch_results = batch_results.view(
        (batch_results.shape[0], batch_results.shape[1], -1))
    for i in range(n_tasks):
        result = torch.nn.functional.cosine_similarity(batch_results[i],
                                                       batch_results[i],
                                                       dim=1)
        result = result.sum()
        correlations[i, i] = correlations[i, i] + result

        for j in np.arange(i + 1, n_tasks):
            result = torch.nn.functional.cosine_similarity(batch_results[i],
                                                           batch_results[j],
                                                           dim=1)
            result = result.sum()
            correlations[i, j] = correlations[i, j] + result
            correlations[j, i] = correlations[j, i] + result

    return correlations


def get_correlation_score_dotprod(batch_results, correlations, drop_version):
    batch_results = torch.stack(batch_results)
    n_tasks = batch_results.shape[0]
    batch_results = batch_results.view(
        (batch_results.shape[0], batch_results.shape[1], -1))
    for i in range(n_tasks):
        result = torch.nn.functional.pairwise_distance(batch_results[i],
                                                       batch_results[i],
                                                       p=1)
        result = result.sum()
        correlations[i, i] = correlations[i, i] + result

        for j in np.arange(i + 1, n_tasks):
            result = torch.nn.functional.pairwise_distance(batch_results[i],
                                                           batch_results[j],
                                                           p=1)
            result = result.sum()
            correlations[i, j] = correlations[i, j] + result
            correlations[j, i] = correlations[j, i] + result

    return correlations


def get_correlation_score(batch_results, correlations, drop_version):
    if drop_version >= 10 and drop_version <= 17:
        return get_correlation_score_ssim(batch_results, correlations,
                                          drop_version)
    elif drop_version == 20 or drop_version == 22:
        return get_correlation_score_cosine(batch_results, correlations,
                                            drop_version)
    elif drop_version == 21 or drop_version == 23:
        return get_correlation_score_dotprod(batch_results, correlations,
                                             drop_version)


def img_for_plot(img):
    '''
    img shape NCHW, ex: torch.Size([3, 1, 256, 256])
    '''
    n, c, _, _ = img.shape
    if c == 2:
        img = img[:, 0:1]
        c = 1

    img_view = img.view(n, c, -1)
    min_img = img_view.min(axis=2)[0][:, :, None, None]
    max_img = img_view.max(axis=2)[0][:, :, None, None]

    return (img - min_img) / (max_img - min_img)


def combine_maps(multi_chan_maps, edges_weights, fct="median"):
    '''
        input list shape: (arr_m1, arr_m2, arr_m3, ...)
        result shape: (arr_all)
    '''
    if len(edges_weights) > 0:
        edges_weights = np.array(edges_weights)
        edges_weights = edges_weights / np.sum(edges_weights)
        edges_weights = edges_weights[:, None, None, None, None]
        edges_weights = torch.tensor(edges_weights, dtype=torch.float32).cuda()
        multi_chan_maps = multi_chan_maps * edges_weights
        return multi_chan_maps.mean(dim=0)
    if fct == "mean":
        return multi_chan_maps.mean(dim=0)
    if fct == "median":
        return median_100(multi_chan_maps)

    assert ('[%s] Combination not implemented' % fct)


def median_100(multi_chan_arr):
    ar_100 = multi_chan_arr * 100.
    ar_100_int = ar_100.int()
    med_100, _ = torch.median(ar_100_int, dim=0)
    return med_100 / 100.


def median_simple(multi_chan_arr):
    med = torch.from_numpy(np.median(multi_chan_arr, axis=0))
    return med


def check_illposed_edge(ensemble_per_sample, edge_per_sample, mean_per_edge,
                        edge, edge_idx, drop_version):
    diff_array = abs(ensemble_per_sample - edge_per_sample)

    # v1, with std
    std = diff_array.std().item()
    if drop_version == 1:
        is_outlier = std > 1

    # v2, with pearson
    r_corr, p_value = pearsonr(ensemble_per_sample, edge_per_sample)
    if drop_version == 2:
        is_outlier = r_corr < 0.9

    # v3
    ensemble_l1 = ensemble_per_sample.mean()
    edge_l1 = edge_per_sample.mean()
    if drop_version == 3:
        is_outlier = (edge_l1 - ensemble_l1) > -0.01

    # v4
    min_value = mean_per_edge.min()
    max_to_take = min_value + (ensemble_l1 - min_value) / 2
    if drop_version == 4:
        is_outlier = edge_l1 - max_to_take > 0

    print(
        "[Start: %19s] std=%3.2f r_corr=%3.2f  (edge: mean=%3.2f)  (ensemble: mean=%3.2f) ::: Outlier?"
        % (edge.expert1.str_id, std, r_corr, edge_l1, ensemble_l1), is_outlier)

    return is_outlier


class DummySummaryWriter:
    def __init__(*args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, *args, **kwargs):
        return self
