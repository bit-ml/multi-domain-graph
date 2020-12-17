import numpy as np
import torch
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim


def get_ssim_score(im1, im2, win_size):
    im1 = np.moveaxis(im1, 0, -1)
    im2 = np.moveaxis(im2, 0, -1)
    ssim_score = ssim(im1, im2, win_size=win_size, multichannel=True)
    return ssim_score


def get_correlation_score(batch_results, correlations, drop_version):
    n_tasks = len(batch_results)
    if drop_version == 10:
        win_size = 255
    elif drop_version == 11:
        win_size = 127
    for i in range(n_tasks):
        for j in range(n_tasks):
            if i > j:
                continue
            task_i_res = batch_results[i]
            task_j_res = batch_results[j]
            scores = []
            for k in range(task_i_res.shape[0]):
                sample_i = task_i_res[k]
                sample_j = task_j_res[k]
                scores.append(get_ssim_score(sample_i, sample_j, win_size))
            sum_scores = np.sum(np.array(scores))
            correlations[i, j] = correlations[i, j] + sum_scores
            correlations[j, i] = correlations[j, i] + sum_scores
    return correlations


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


def combine_maps(result_list, fct="median"):
    '''
        input list shape: (arr_m1, arr_m2, arr_m3, ...)
        result shape: (arr_all)
    '''
    multi_chan_arr = np.array(result_list)
    if fct == "mean":
        return torch.from_numpy(multi_chan_arr.mean(axis=0))
    if fct == "median":
        return median_100(multi_chan_arr)

    assert ('[%s] Combination not implemented' % fct)


def median_100(multi_chan_arr):
    ar_100 = multi_chan_arr * 100.
    ar_100_int = ar_100.astype(np.int32)
    med_100 = np.median(ar_100_int, axis=0).astype(np.float32)
    med_100_th = torch.from_numpy(med_100) / 100.
    return med_100_th


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
