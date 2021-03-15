import glob

import numpy as np
import torch
from torch_two_sample.statistics_diff import MMDStatistic

from experts.depth_expert import DepthModelXTC

# from torch_two_sample.util import pdist

DATASETS = [("replica", "test"), ("taskonomy", "tiny-test"),
            ("hypersim", "test")]

n_points = 1000
embeding_size = 3


n_points = 16
embeding_size = 1024

num_files_to_compare = 10


def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2)**norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner)**(1. / norm)


def my_mmd(sample_1, sample_2):
    sample_12 = torch.cat((sample_1, sample_2), 0)
    distances = pdist(sample_12, sample_12, norm=2)
    kernels = None
    for alpha in [1.]:
        kernels_a = torch.exp(-alpha * distances**2)
        if kernels is None:
            kernels = kernels_a
        else:
            kernels = kernels + kernels_a
    k_1 = kernels[:mmd.n_1, :mmd.n_1]
    k_2 = kernels[mmd.n_1:, mmd.n_1:]
    k_12 = kernels[:mmd.n_1, mmd.n_1:]

    score = (2 * mmd.a01 * k_12.sum() + mmd.a00 *
             (k_1.sum() - torch.trace(k_1)) + mmd.a11 *
             (k_2.sum() - torch.trace(k_2)))
    return score


depth_expert = DepthModelXTC(full_expert=True)
depth_expert.model.eval()

for DATASET1, SPLIT1 in DATASETS:
    for DATASET2, SPLIT2 in DATASETS:
        PATH_FILES1 = "/data/multi-domain-graph-2/datasets/datasets_preproc_gt/%s/%s/rgb/" % (
            DATASET1, SPLIT1)
        PATH_FILES2 = "/data/multi-domain-graph-2/datasets/datasets_preproc_gt/%s/%s/rgb/" % (
            DATASET2, SPLIT2)

        files1 = sorted(glob.glob(PATH_FILES1 +
                                  "/*.npy"))[:num_files_to_compare]
        files2 = sorted(glob.glob(PATH_FILES2 +
                                  "/*.npy"))[:num_files_to_compare]

        mmd = MMDStatistic(n_points, n_points)

        mmd_scores = []
        for idx1, sample1_path in enumerate(files1):
            for idx2, sample2_path in enumerate(files2):
                torchinp1 = torch.from_numpy(np.load(sample1_path))
                torchinp2 = torch.from_numpy(np.load(sample2_path))

                # v2. compare RGB
                # sample_1 = torchinp1.permute(1, 2, 0).view((-1, embeding_size))[:n_points, :]
                # sample_2 = torchinp2.permute(1, 2, 0).view((-1, embeding_size))[:n_points, :]

                # v2. compare expert
                with torch.no_grad():
                    torchinp1 = torchinp1.cuda()
                    torchinp2 = torchinp2.cuda()
                    _, x_mid_layer1 = depth_expert.model(torchinp1[None])
                    _, x_mid_layer2 = depth_expert.model(torchinp2[None])
                    sample_1 = x_mid_layer1[0].permute(1, 2, 0).view(
                        n_points, embeding_size)
                    sample_2 = x_mid_layer2[0].permute(1, 2, 0).view(
                        n_points, embeding_size)

                score = mmd(sample_1, sample_2, alphas=[1.])
                mmd_scores.append(score.data.cpu().numpy())
        print("[%15s - %15s] mmd = %.2f" %
              (DATASET1, DATASET2, np.mean(mmd_scores) * 100.))
    print("")
