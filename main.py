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
from utils.utils import DummySummaryWriter

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


############################## drop connections ###############################
def drop_connections(space_graph, drop_version):
    if drop_version > 0:
        if drop_version < 10:
            drop_connections_simple(space_graph, drop_version)
        else:
            drop_connections_correlations(space_graph, drop_version)


def drop_connections_correlations(space_graph, drop_version):
    for expert_idx, expert in enumerate(space_graph.experts.methods):
        #print('=======')
        #print('EXPERT %s' % expert.str_id)
        #print('=======')

        # get edges ending in current expert
        ending_edges = []
        ending_edges_src_identifiers = []
        for edge in space_graph.edges:
            if edge.expert2.identifier == expert.identifier:
                ending_edges.append(edge)
                ending_edges_src_identifiers.append(edge.expert1.identifier)
        ending_edges_src_identifiers.append(expert.identifier)

        expert_correlations = Edge.drop_1hop_connections(
            ending_edges, device, drop_version).cpu().numpy()

        if drop_version == 10 or drop_version == 11 or drop_version == 14 or drop_version == 15 or drop_version == 20 or drop_version == 21:
            import numpy.linalg as linalg
            eigenValues, eigenVectors = linalg.eig(expert_correlations)
            pos = np.argmax(eigenValues)
            task_weights = eigenVectors[:, pos]
        elif drop_version == 12 or drop_version == 13 or drop_version == 16 or drop_version == 17 or drop_version == 22 or drop_version == 23:
            task_weights = expert_correlations[-1, :]
        #print(task_weights)

        min_val = np.min(task_weights)
        max_val = np.max(task_weights)

        task_weights = (task_weights - min_val) / (max_val - min_val)
        #print(task_weights)

        if drop_version == 12 or drop_version == 13 or drop_version == 22 or drop_version == 23:
            task_weights[task_weights <= 0] = 0
        elif drop_version == 10 or drop_version == 11 or drop_version == 20 or drop_version == 21:  # or drop_version == 14 or drop_version == 15:
            task_weights[task_weights < 0.5] = 0
        elif drop_version == 14 or drop_version == 15 or drop_version == 16 or drop_version == 17:
            task_weights[task_weights == 0] = 0.01

        remove_indexes = np.argwhere(task_weights == 0)[:, 0]
        keep_indexes = np.argwhere(task_weights > 0)[:, 0]

        remove_indexes = remove_indexes[remove_indexes < len(ending_edges)]
        keep_indexes = keep_indexes[keep_indexes < len(ending_edges)]

        if drop_version == 10 or drop_version == 11 or drop_version == 12 or drop_version == 13 or drop_version == 20 or drop_version == 21 or drop_version == 22 or drop_version == 23:
            for idx in remove_indexes:
                #print("remove edge from: %s"%(ending_edges[idx].expert1.str_id))
                ending_edges[idx].ill_posed = True
        for ending_edge in ending_edges:
            ending_edge.in_edge_weights = task_weights
            ending_edge.in_edge_src_identifiers = ending_edges_src_identifiers


def drop_connections_simple(space_graph, drop_version):
    # for each node domain (= expert in our case)
    for expert_idx, expert in enumerate(space_graph.experts.methods):
        ending_edges = []
        # 1. List all edges reaching this node
        for edge in space_graph.edges:
            if edge.expert2.str_id == expert.str_id:
                ending_edges.append(edge)

        l1_per_edge_per_sample = torch.from_numpy(
            evaluate_all_edges(ending_edges))

        # 2. Check ensembles value vs single edge
        ensemble_l1_per_sample = utils.combine_maps(l1_per_edge_per_sample,
                                                    fct="median")
        mean_l1_per_edge = l1_per_edge_per_sample.mean(dim=1)

        print("\n==== End node [%19s] =====" % expert.str_id)
        for edge_idx, edge in enumerate(ending_edges):
            is_outlier = utils.check_illposed_edge(
                ensemble_l1_per_sample, l1_per_edge_per_sample[edge_idx],
                mean_l1_per_edge, edge, edge_idx, drop_version)
            edge.ill_posed = is_outlier
        print("============================")


############################## 1HOP ###############################
def eval_1hop_ensembles(space_graph, drop_version, silent, config):
    csv_path = os.path.join(config.get('Logs', 'csv_results_dir'))
    if not os.path.exists(csv_path):
        os.mkdir(csv_path)
    csv_path = os.path.join(
        csv_path, 'results_%s.csv' % (config.get('Run id', 'datetime')))
    if drop_version == -1:
        csv_file = open(csv_path, 'w')
        csv_file.write('metric: L1\n')
        csv_file.write(',')
        for expert in space_graph.experts.methods:
            csv_file.write('to_%s,,' % (expert.identifier))
        csv_file.write('\n')
        csv_file.write(',')
        for expert in space_graph.experts.methods:
            csv_file.write('valid,test,')
        csv_file.write('\n')
        csv_file.write('before drop,')
        csv_file.close()
    else:
        csv_file = open(csv_path, 'a')
        csv_file.write('after drop,')
        csv_file.close()

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
    save_idxes_test = None

    for expert in space_graph.experts.methods:
        end_id = expert.identifier
        tag = "Valid_1Hop_%s" % end_id
        edges_1hop = []
        edges_1hop_weights = []
        edges_1hop_test_weights = []

        # 1. Select edges that ends in end_id
        for edge_xk in space_graph.edges:
            if edge_xk.ill_posed:
                continue
            if edge_xk.expert2.identifier == end_id:
                edges_1hop.append(edge_xk)
                edge_weights = edge_xk.in_edge_weights
                edge_src_identifiers = edge_xk.in_edge_src_identifiers
                if drop_version == 14 or drop_version == 15 or drop_version == 16 or drop_version == 17 or drop_version == 22 or drop_version == 23:
                    index = edge_src_identifiers.index(
                        edge_xk.expert1.identifier)
                    edges_1hop_weights.append(edge_weights[index])
                    if edge_xk.test_loader != None:
                        edges_1hop_test_weights.append(edge_weights[index])

        if drop_version == 14 or drop_version == 15 or drop_version == 16 or drop_version == 17 or drop_version == 22 or drop_version == 23:
            # add weight of the expert pseudo gt
            edges_1hop_weights.append(edge_weights[-1])
            edges_1hop_test_weights.append(edge_weights[-1])

        # 2. Eval each ensemble
        if len(edges_1hop) > 0:
            save_idxes, save_idxes_test = Edge.eval_1hop_ensemble(
                edges_1hop, save_idxes, save_idxes_test, device, writer,
                drop_version, csv_path, edges_1hop_weights,
                edges_1hop_test_weights)

    writer.close()

    csv_file = open(csv_path, 'a')
    csv_file.write('\n')
    csv_file.close()


def train_2Dtasks(space_graph, epochs, silent, config):
    if silent:
        writer = DummySummaryWriter()
    else:
        tb_dir = config.get('Logs', 'tensorboard_dir')
        tb_prefix = config.get('Logs', 'tensorboard_prefix')

        writer = SummaryWriter(log_dir=f'%s/%s_train_2Dtasks_%s' %
                               (tb_dir, tb_prefix, datetime.now()),
                               flush_secs=30)
    eval_test = config.getboolean('Training', 'eval_test_during_train')

    src_domain_restr = config.get('Training', 'src_domain_restr')

    for net_idx, net in enumerate(space_graph.edges):
        if config.getboolean(
                'Training', 'restr_src_domain'
        ) and not net.expert1.domain_name == src_domain_restr:
            continue
        print("[%2d] Train" % net_idx, net)

        net.train(epochs=epochs,
                  device=device,
                  writer=writer,
                  eval_test=eval_test)

    writer.close()


def load_2Dtasks(graph, epoch):
    print("Load nets from checkpoints")
    for net_idx, edge in enumerate(graph.edges):
        path = os.path.join(edge.load_model_dir, 'epoch_%05d.pth' % (epoch))
        edge.net.load_state_dict(torch.load(path))
        edge.net.module.eval()


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

    print("Eval 1Hop ensembles before drop")
    # drop_version passed as -1 -> no drop
    eval_1hop_ensembles(graph, drop_version=-1, silent=silent, config=config)

    # 3. Drop ill-posed connections
    drop_version = config.getint('Training', 'drop_version')
    drop_connections(graph, drop_version)

    # 4. Eval 1Hop
    print("Eval 1Hop ensembles after drop (version %i)" % drop_version)
    drop_version = config.getint('Training', 'drop_version')
    eval_1hop_ensembles(graph, drop_version, silent=silent, config=config)

    # # 5. Train/Eval 2Hop
    # print("Eval 2Hop ensembles")
    # # used only as eval
    # train_2hops_2Dtasks(graph,
    #                     drop_version,
    #                     epochs=1,
    #                     use_expert_gt=True,
    #                     silent=silent,
    #                     config=config)


if __name__ == "__main__":
    main(sys.argv)
