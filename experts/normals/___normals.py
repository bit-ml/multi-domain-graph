import torch
# import os
from networks.gnn import GNNFixedK
from torch_sym3eig import Sym3Eig
from utils.covariance import (compute_cov_matrices_dense,
                              compute_weighted_cov_matrices_dense)
# import os.path as osp
# import numpy as np
# import torch
# import torch_geometric.transforms as T
# import argparse
# from datasets.pcpnet_dataset import PCPNetDataset
# from torch_geometric.data import DataLoader
from utils.radius import radius_graph


class NormalEstimation(torch.nn.Module):
    def __init__(self):
        super(NormalEstimation, self).__init__()
        self.stepWeights = GNNFixedK()

    def forward(self, old_weights, pos, normals, edge_idx_l, dense_l, stddev):
        # Re-weighting
        weights = self.stepWeights(pos, old_weights, normals, edge_idx_l,
                                   dense_l, stddev)  # , f=f)

        # Weighted Least-Squares
        cov = compute_weighted_cov_matrices_dense(pos, weights, dense_l,
                                                  edge_idx_l[0])
        eig_val, eig_vec = Sym3Eig.apply(cov)
        _, argsort = torch.abs(eig_val).sort(dim=-1, descending=False)
        eig_vec = eig_vec.gather(2, argsort.view(-1, 1, 3).expand_as(eig_vec))
        normals = eig_vec[:, :, 0]

        # Not necessary for PCPNetDataset but might be for other datasets with underdefined neighborhoods
        # mask = torch.isnan(normals)
        # normals[mask] = 0.0

        return normals, weights


def test(model, iters, k_size, pos, batch):
    with torch.no_grad():
        # Compute statistics for normalization
        edge_idx_16, _ = radius_graph(pos,
                                      0.5,
                                      batch=batch,
                                      max_num_neighbors=16)
        row16, col16 = edge_idx_16
        cart16 = (pos[col16].cuda() - pos[row16].cuda())
        stddev = torch.sqrt((cart16**2).mean()).detach().item()

        # Compute KNN-graph indices for GNN
        edge_idx_l, dense_l = radius_graph(pos,
                                           0.5,
                                           batch=batch,
                                           max_num_neighbors=k_size)

        # Iteration 0 (PCA)
        cov = compute_cov_matrices_dense(pos, dense_l, edge_idx_l[0]).cuda()
        eig_val, eig_vec = Sym3Eig.apply(cov)
        _, argsort = torch.abs(eig_val).sort(dim=-1, descending=False)
        eig_vec = eig_vec.gather(2, argsort.view(-1, 1, 3).expand_as(eig_vec))
        # mask = torch.isnan(eig_vec)
        # eig_vec[mask] = 0.0
        normals = eig_vec[:, :, 0]
        edge_idx_c = edge_idx_l.cuda()
        pos, batch = pos.detach().cuda(), batch.detach().cuda()
        old_weights = torch.ones_like(edge_idx_c[0]).float() / float(k_size)

        # Loop of Algorithm 1 in the paper
        for j in range(iters):
            normals, old_weights = model(old_weights.detach(), pos,
                                         normals.detach(), edge_idx_c,
                                         edge_idx_c[1].view(pos.size(0),
                                                            -1), stddev)

    # if FLAGS.results_path is not None:
    #     save_normals(normals.detach(), test_set, i)

    return normals.detach()
