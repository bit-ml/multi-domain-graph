import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from experts.experts import Experts
from experts.save_output import (generate_experts_output,
                                 generate_experts_output_with_time)
from graph.edges.graph_edges import Edge
from graph.graph import MultiDomainGraph
from utils import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import configparser


def build_space_graph(config, silent, valid_shuffle):
    all_experts = Experts(full_experts=False)
    md_graph = MultiDomainGraph(config,
                                all_experts,
                                device,
                                silent=silent,
                                valid_shuffle=valid_shuffle)
    return md_graph


def evaluate_all_edges(ending_edges):
    metrics = []
    for edge in ending_edges:
        edge_l2_loss, edge_l1_loss = edge.eval_detailed(device)
        metrics.append(np.array(edge_l1_loss))

    return np.array(metrics)


def drop_connections(space_graph, drop_version):
    # for each node domain (= expert in our case)
    for expert_idx, expert in enumerate(space_graph.experts.methods):
        ending_edges = []
        # 1. List all edges reaching this node
        for edge in space_graph.edges:
            if edge.expert2.str_id == expert.str_id:
                ending_edges.append(edge)

        l1_per_edge_per_sample = evaluate_all_edges(ending_edges)

        # 2. Check ensembles value vs single edge
        ensemble_l1_per_sample = utils.combine_maps(l1_per_edge_per_sample,
                                                    fct="median").numpy()
        mean_l1_per_edge = l1_per_edge_per_sample.mean(axis=1)

        print("\n==== End node [%19s] =====" % expert.str_id)
        for edge_idx, edge in enumerate(ending_edges):
            is_outlier = utils.check_illposed_edge(
                ensemble_l1_per_sample, l1_per_edge_per_sample[edge_idx],
                mean_l1_per_edge, edge, edge_idx, drop_version)
            edge.ill_posed = is_outlier
        print("============================")


############################## 1HOP ###############################
# TODO: add use_expert_gt parameter
def eval_1hop_ensembles(space_graph, drop_version, silent, config):
    if silent:
        writer = DummySummaryWriter()
    else:
        tb_dir = config.get('Logs', 'tensorboard_dir')
        tb_prefix = config.get('Logs', 'tensorboard_prefix')

        writer = SummaryWriter(
            log_dir=f'%s/%s_1hop_ens_dropV%d_%s' %
            (tb_dir, tb_prefix, drop_version, datetime.now()),
            flush_secs=30)
    save_idxes = None

    for expert in space_graph.experts.methods:
        end_id = expert.str_id
        tag = "Valid_1Hop_%s" % end_id
        edges_1hop = []

        # 1. Select edges that ends in end_id
        for edge_xk in space_graph.edges:
            if edge_xk.ill_posed:
                continue
            if edge_xk.expert2.str_id == end_id:
                edges_1hop.append(edge_xk)

        # 2. Eval each ensemble
        save_idxes = Edge.eval_1hop_ensemble(edges_1hop, save_idxes, device,
                                             writer)

    writer.close()


def train_2Dtasks(space_graph, epochs, silent, config):
    if silent:
        writer = DummySummaryWriter()
    else:
        tb_dir = config.get('Logs', 'tensorboard_dir')
        tb_prefix = config.get('Logs', 'tensorboard_prefix')

        writer = SummaryWriter(log_dir=f'%s/%s_train_2Dtasks_%s' %
                               (tb_dir, tb_prefix, datetime.now()),
                               flush_secs=30)

    for net_idx, net in enumerate(space_graph.edges):
        print("[%2d] Train" % net_idx, net)
        net.train(epochs=epochs, device=device, writer=writer)

    writer.close()


def load_2Dtasks(graph, epoch):
    print("Load nets from checkpoints")
    for net_idx, edge in enumerate(graph.edges):
        path = os.path.join(edge.load_model_dir, 'epoch_%05d.pth' % (epoch))
        edge.net.load_state_dict(torch.load(path))
        edge.net.eval()


############################## 2HOPs ###############################


def train_2hops_2Dtasks(space_graph, drop_version, epochs, use_expert_gt,
                        silent, config):
    '''
    - Like train_1hop_2Dtasks, but use ensembles as GT.
    - Use bool: edge.ill_posed
    '''
    if silent:
        writer = DummySummaryWriter()
    else:
        tb_dir = config.get('Logs', 'tensorboard_dir')
        tb_prefix = config.get('Logs', 'tensorboard_prefix')

        writer = SummaryWriter(
            log_dir=f'%s/%s_2hops_ens_dropV%d_%s' %
            (tb_dir, tb_prefix, drop_version, datetime.now()),
            flush_secs=30)

    for net_idx, net in enumerate(space_graph.edges):
        print("[net_idx %2d] Train with 2-hop supervision" % (net_idx), net)
        net.train_from_2hops_ens(graph=space_graph,
                                 epochs=epochs,
                                 drop_version=drop_version,
                                 device=device,
                                 writer=writer,
                                 use_expert_gt=use_expert_gt)
    writer.close()


############################## MAIN ###############################


def main(argv):
    config = configparser.ConfigParser()
    config.read(argv[1])
    config.set('Run id', 'datetime', str(datetime.now()))

    silent = config.getboolean('Logs', 'silent')
    epochs = config.getint('Edge Models', 'n_epochs')
    # 0. Generate experts output
    if config.getboolean('Preprocess', 'generate_experts_output'):
        generate_experts_output(Experts(full_experts=True).methods, config)
        sys.exit(0)

    if config.getboolean('Training', 'train_basic_edges'):
        # 1. Build graph + Train 1Hop
        graph = build_space_graph(config, silent=silent, valid_shuffle=True)
        train_2Dtasks(graph, epochs=epochs, silent=silent, config=config)
        sys.exit(0)
    else:
        # 2. Build graph + Load 1Hop edges
        graph = build_space_graph(config, silent=silent, valid_shuffle=False)
        load_2Dtasks(graph, epoch=epochs)
    '''
    # 3. Drop ill-posed connections
    drop_version = config.getint('Training', 'drop_version')
    if drop_version > 0:
        # TODO: integrate solution from Ema
        drop_connections(graph, drop_version)

    # 4. Eval 1Hop
    print("Eval 1Hop ensembles")
    eval_1hop_ensembles(graph, drop_version, silent=silent, config=config)

    # 5. Train/Eval 2Hop
    print("Eval 2Hop ensembles")
    # used only as eval
    train_2hops_2Dtasks(graph,
                        drop_version,
                        epochs=1,
                        use_expert_gt=True,
                        silent=silent,
                        config=config)
    '''

if __name__ == "__main__":
    main(sys.argv)
