import os
import pathlib
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from experts.experts import Experts
from graph.edges.graph_edges import Edge
from graph.graph import MultiDomainGraph
from utils import utils
from utils.utils import DummySummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import configparser


def build_space_graph(config, silent, valid_shuffle, iter_no=1):
    use_rgb_to_tsk = config.getboolean('Ensemble', 'use_rgb_to_tsk')

    if config.has_option('Experts', 'selector_map'):
        selector_map_str = config.get('Experts', 'selector_map').split(",")
        selector_map = [int(token) for token in selector_map_str]
    else:
        selector_map = None

    all_experts = Experts(dataset_name=config.get('Paths', 'DATASET_NAME'),
                          full_experts=False,
                          use_rgb_to_tsk=use_rgb_to_tsk,
                          selector_map=selector_map)

    md_graph = MultiDomainGraph(
        config,
        all_experts,
        device,
        iter_no=iter_no,
        silent=silent,
        valid_shuffle=valid_shuffle,
    )
    return md_graph


def evaluate_all_edges(ending_edges):
    metrics = []
    for edge in ending_edges:
        edge_l2_loss, edge_l1_loss = edge.eval_detailed(device)
        metrics.append(np.array(edge_l1_loss))

    return np.array(metrics)


def eval_1hop(space_graph, silent, config, epoch_idx, iter_idx):
    csv_results_path = config.get('Logs', 'csv_results')
    if not os.path.exists(csv_results_path):
        os.mkdir(csv_results_path)

    valid_dataset = config.get('PathsIter%d' % iter_idx,
                               'ITER%d_VALID_SRC_PATH' % iter_idx)
    valid_set_str = pathlib.Path(valid_dataset).parts[-1]

    test_dataset = config.get('PathsIter%d' % iter_idx,
                              'ITER%d_TEST_SRC_PATH' % iter_idx)
    test_set_str = pathlib.Path(test_dataset).parts[-1]

    if silent:
        writer = DummySummaryWriter()
    else:
        tb_dir = config.get('Logs', 'tensorboard_dir')
        tb_prefix = config.get('Logs', 'tensorboard_prefix')
        datetime = config.get('Run id', 'datetime')
        writer = SummaryWriter(
            log_dir=f'%s/%s_1hop_edges_e%d_valid_%s_test_%s_%s' %
            (tb_dir, tb_prefix, epoch_idx, valid_set_str, test_set_str,
             datetime),
            flush_secs=30)
    save_idxes = None
    save_idxes_test = None

    for expert in space_graph.experts.methods:
        end_id = expert.identifier

        edges_1hop = []

        # 1. Select edges that ends in end_id
        for edge_xk in space_graph.edges:
            if edge_xk.ill_posed:
                continue
            if not edge_xk.trained:
                continue
            if edge_xk.expert2.identifier == end_id:
                edges_1hop.append(edge_xk)

        # 2. Eval all edges towards end_id
        if len(edges_1hop) > 0:
            save_idxes, save_idxes_test = Edge.eval_1hop(
                edges_1hop, save_idxes, save_idxes_test, device, writer,
                valid_set_str, test_set_str, csv_results_path, epoch_idx)
        else:
            print('--------dst: %s === NOT TESTED' % end_id)

    writer.close()


############################## drop connections ###############################
def drop_connections(space_graph, drop_version):
    if drop_version > 0:
        if drop_version < 10:
            drop_connections_simple(space_graph, drop_version)
        else:
            drop_connections_correlations(space_graph, drop_version)


def drop_connections_correlations(space_graph, drop_version):
    for expert_idx, expert in enumerate(space_graph.experts.methods):
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

        if drop_version == 21 or drop_version == 23:
            min_v = np.min(expert_correlations)
            max_v = np.max(expert_correlations)
            expert_correlations = (expert_correlations - min_v) / (max_v -
                                                                   min_v)
            expert_correlations = 1 - expert_correlations
        if drop_version == 10 or drop_version == 11 or drop_version == 14 or drop_version == 15 or drop_version == 20 or drop_version == 21:
            import numpy.linalg as linalg
            eigenValues, eigenVectors = linalg.eig(expert_correlations)
            '''
            s_indexes = np.flip(np.argsort(eigenValues))
            eigenVectors = eigenVectors[s_indexes, :]
            min_v = np.min(eigenVectors, 0)[None, :]
            max_v = np.max(eigenVectors, 0)[None, :]
            diff = max_v - min_v
            diff[diff < 0.0000001] = 1
            eigenVectors = (eigenVectors - min_v) / (max_v - min_v)
            cl_pos = np.argwhere(eigenVectors[-1, :] >= 0.5)
            if len(cl_pos) > 0:
                task_weights = eigenVectors[:, cl_pos[0]]
            else:
                task_weights = np.ones((eigenVectors.shape[0], ))
            '''
            pos = np.argmax(eigenValues)
            task_weights = eigenVectors[:, pos]
        elif drop_version == 12 or drop_version == 13 or drop_version == 16 or drop_version == 17 or drop_version == 22 or drop_version == 23:
            task_weights = expert_correlations[-1, :]

        min_val = np.min(task_weights)
        max_val = np.max(task_weights)

        task_weights = (task_weights - min_val) / (max_val - min_val)

        if drop_version == 12 or drop_version == 13 or drop_version == 22 or drop_version == 23:
            indexes = np.argsort(task_weights)
            indexes = indexes[0:len(task_weights) - 4]
            task_weights[indexes] = 0
            #task_weights[task_weights <= 0] = 0
        elif drop_version == 10 or drop_version == 11 or drop_version == 20 or drop_version == 21:  # or drop_version == 14 or drop_version == 15:
            # if current task is not part of the main cluster => abort
            #if task_weights[-1] >= 0.5:
            task_weights[task_weights < 0.5] = 0
            #else:
            #    task_weights[task_weights == 0] = 0.0001
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

        print('EXPERT: %20s' % expert.identifier)
        for i in range(len(ending_edges_src_identifiers)):
            if i < len(ending_edges) and ending_edges[i].ill_posed:
                drop_str = 'drop'
            else:
                drop_str = ''
            print('--%20s %20.10f - %s' %
                  (ending_edges_src_identifiers[i], task_weights[i], drop_str))


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
    if silent:
        writer = DummySummaryWriter()
    else:
        tb_dir = config.get('Logs', 'tensorboard_dir')
        tb_prefix = config.get('Logs', 'tensorboard_prefix')
        datetime = config.get('Run id', 'datetime')
        writer = SummaryWriter(log_dir=f'%s/%s_1hop_ens_dropV%d_%s' %
                               (tb_dir, tb_prefix, drop_version, datetime),
                               flush_secs=30)
    save_idxes = None
    save_idxes_test = None
    add_rgb_src_in_ensemble = config.getboolean('Ensemble',
                                                'add_rgb_src_in_ensemble')

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
            if not edge_xk.trained:
                continue
            if not add_rgb_src_in_ensemble and edge_xk.expert1.domain_name == 'rgb':
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
                drop_version, edges_1hop_weights, edges_1hop_test_weights,
                config)

    writer.close()


def save_1hop_ensembles(space_graph, config, iter_no):
    writer = DummySummaryWriter()

    add_rgb_src_in_ensemble = config.getboolean('Ensemble',
                                                'add_rgb_src_in_ensemble')

    for expert in space_graph.experts.methods:
        end_id = expert.identifier
        edges_1hop = []

        # 1. Select edges that ends in end_id
        for edge_xk in space_graph.edges:
            if edge_xk.ill_posed:
                continue
            if not edge_xk.trained:
                continue
            if not add_rgb_src_in_ensemble and edge_xk.expert1.domain_name == 'rgb':
                continue
            if edge_xk.expert2.identifier == end_id:
                edges_1hop.append(edge_xk)

        # 2. Eval each ensemble
        if len(edges_1hop) > 0:
            Edge.save_1hop_ensemble(edges_1hop, device, config, iter_no)

    writer.close()


def train_2Dtasks(space_graph, start_epoch, n_epochs, silent, config):
    if silent:
        writer = DummySummaryWriter()
    else:
        tb_dir = config.get('Logs', 'tensorboard_dir')
        tb_prefix = config.get('Logs', 'tensorboard_prefix')
        datetime = config.get('Run id', 'datetime')
        writer = SummaryWriter(
            log_dir=f'%s/%s_train_2Dtasks_%s' %
            (tb_dir, tb_prefix, datetime),  #datetime.now()),
            flush_secs=30)
    eval_test = config.getboolean('Training', 'eval_test_during_train')

    src_domain_restr = config.get('Training', 'src_domain_restr')

    for net_idx, net in enumerate(space_graph.edges):
        if config.getboolean(
                'Training', 'restr_src_domain'
        ) and not net.expert1.domain_name == src_domain_restr:
            continue
        print("[%2d] Train" % net_idx, net)

        net.train(start_epoch=start_epoch,
                  n_epochs=n_epochs,
                  device=device,
                  writer=writer,
                  eval_test=eval_test)
        net.trained = True

    writer.close()


def check_models_exists(config, epoch):
    print("Load nets from checkpoints. From epoch: %2d" % epoch,
          config.get('Edge Models', 'load_path'))
    all_experts = Experts(full_experts=False)
    for expert_i in all_experts.methods:
        for expert_j in all_experts.methods:
            if expert_i != expert_j:
                load_path = os.path.join(
                    config.get('Edge Models',
                               'load_path'), '%s_%s/epoch_%05d.pth' %
                    (expert_i.identifier, expert_j.identifier, epoch))
                if not os.path.exists(load_path):
                    print("NU Exista: %15s ---> %15s (epoch %d)" %
                          (expert_i.identifier, expert_j.identifier, epoch))


def load_2Dtasks(graph, epoch):
    print("Load nets from checkpoints. From epoch: %2d" % epoch)
    for net_idx, edge in enumerate(graph.edges):
        path = os.path.join(edge.load_model_dir, 'epoch_%05d.pth' % (epoch))
        if os.path.exists(path):
            edge.net.load_state_dict(torch.load(path))
            edge.net.module.eval()
            edge.trained = True
        else:
            print(
                'model: %s_%s UNAVAILABLE' %
                (edge.expert1.domain_name, edge.expert2.domain_name), path)


############################## MAIN ###############################
def preprocess_config_file_paths(config):
    n_iters = config.getint('General', 'n_iters')
    splits = ['TRAIN', 'TEST', 'VALID']
    # complete possible None values
    for i in np.arange(2, n_iters + 1):
        for split in splits:
            SRC_PATH = config.get('PathsIter%d' % i,
                                  'ITER%d_%s_SRC_PATH' % (i, split))
            DST_PATH = config.get('PathsIter%d' % i,
                                  'ITER%d_%s_DST_PATH' % (i, split))
            PATTERNS = config.get('PathsIter%d' % i,
                                  'ITER%d_%s_PATTERNS' % (i, split))
            FIRST_K = config.get('PathsIter%d' % i,
                                 'ITER%d_%s_FIRST_K' % (i, split))
            if split == 'TEST':
                GT_DST_PATH = config.get('PathsIter%d' % i,
                                         'ITER%d_%s_GT_DST_PATH' % (i, split))
            else:
                GT_DST_PATH = DST_PATH
            assert ((SRC_PATH == '' and DST_PATH == '' and PATTERNS == ''
                     and FIRST_K == '' and GT_DST_PATH == '')
                    or ((not SRC_PATH == '') and (not DST_PATH == '') and
                        (not PATTERNS == '') and (not FIRST_K == '') and
                        (not GT_DST_PATH == '')))
            if SRC_PATH == '':
                config.set(
                    'PathsIter%d' % i, 'ITER%d_%s_SRC_PATH' % (i, split),
                    config.get('PathsIter1', 'ITER1_%s_SRC_PATH' % split))
                config.set(
                    'PathsIter%d' % i, 'ITER%d_%s_DST_PATH' % (i, split),
                    config.get('PathsIter1', 'ITER1_%s_DST_PATH' % split))
                config.set(
                    'PathsIter%d' % i, 'ITER%d_%s_FIRST_K' % (i, split),
                    config.get('PathsIter1', 'ITER1_%s_FIRST_K' % split))
                config.set(
                    'PathsIter%d' % i, 'ITER%d_%s_PATTERNS' % (i, split),
                    config.get('PathsIter1', 'ITER1_%s_PATTERNS' % split))
                if split == 'TEST':
                    config.set(
                        'PathsIter%d' % i,
                        'ITER%d_%s_GT_DST_PATH' % (i, split),
                        config.get('PathsIter1',
                                   'ITER1_%s_GT_DST_PATH' % split))

    # check consistency
    for i in np.arange(1, n_iters + 1):
        for split in splits:
            SRC_PATH = config.get('PathsIter%d' % i, 'ITER%d_%s_SRC_PATH' %
                                  (i, split)).split('\n')
            DST_PATH = config.get('PathsIter%d' % i, 'ITER%d_%s_DST_PATH' %
                                  (i, split)).split('\n')
            PATTERNS = config.get('PathsIter%d' % i, 'ITER%d_%s_PATTERNS' %
                                  (i, split)).split('\n')
            FIRST_K = config.getint('PathsIter%d' % i,
                                    'ITER%d_%s_FIRST_K' % (i, split))
            if split == 'TEST':
                GT_DST_PATH = config.get('PathsIter%d' % i,
                                         'ITER%d_%s_GT_DST_PATH' %
                                         (i, split)).split('\n')
            assert (len(SRC_PATH) == len(DST_PATH))
            if split == 'Test':
                assert (len(SRC_PATH) == len(GT_DST_PATH))
            if i > 1:
                STORE_PATH = config.get('PathsIter%d' % i,
                                        'ITER%d_%s_STORE_PATH' %
                                        (i, split)).split('\n')
                assert (len(STORE_PATH) == len(SRC_PATH))


def preprocess_config_file(config):
    '''
        - s.t. we reduce branching during the actual run 
        e.g. if ITER2 paths are not set -> will be set to paths of ITER1, except for the store paths 
    '''
    # set 'Run id'
    config.set('Run id', 'datetime', str(datetime.now()))
    # set paths for iters
    preprocess_config_file_paths(config)


def prepare_store_folders(config, iter_no, all_experts):
    next_iter_train_store_path = config.get(
        'PathsIter%d' % iter_no,
        'ITER%d_TRAIN_STORE_PATH' % iter_no).split('\n')
    next_iter_valid_store_path = config.get(
        'PathsIter%d' % iter_no,
        'ITER%d_VALID_STORE_PATH' % iter_no).split('\n')
    next_iter_test_store_path = config.get('PathsIter%d' % iter_no,
                                           'ITER%d_TEST_STORE_PATH' %
                                           iter_no).split('\n')
    for expert in all_experts.methods:
        for i in range(len(next_iter_train_store_path)):
            save_to_dir = "%s/%s" % (next_iter_train_store_path[i],
                                     expert.identifier)
            os.makedirs(save_to_dir, exist_ok=True)
        for i in range(len(next_iter_valid_store_path)):
            save_to_dir = "%s/%s" % (next_iter_valid_store_path[i],
                                     expert.identifier)
            os.makedirs(save_to_dir, exist_ok=True)
        for i in range(len(next_iter_test_store_path)):
            save_to_dir = "%s/%s" % (next_iter_test_store_path[i],
                                     expert.identifier)
            os.makedirs(save_to_dir, exist_ok=True)


def main(argv):
    config = configparser.ConfigParser()
    config.read(argv[1])
    preprocess_config_file(config)

    print(config.get('Run id', 'datetime'))
    print("load_path", config.get('Edge Models', 'load_path'))

    silent = config.getboolean('Logs', 'silent')
    n_epochs = config.getint('Edge Models', 'n_epochs')
    start_epoch = config.getint('Edge Models', 'start_epoch')

    # Eval 1hop models
    if config.getboolean('Testing', 'test_1hop_edges'):
        min_epoch = config.getint('Testing', 'test_min_epoch')
        max_epoch = config.getint('Testing', 'test_max_epoch')
        epoch_step = max(1, config.getint('Testing', 'test_epoch_step'))
        for t_epoch in np.arange(min_epoch, max_epoch + 1, epoch_step):
            graph = build_space_graph(config,
                                      silent=silent,
                                      valid_shuffle=False)
            load_2Dtasks(graph, epoch=t_epoch)
            eval_1hop(graph,
                      silent=silent,
                      config=config,
                      epoch_idx=t_epoch,
                      iter_idx=1)
            #eval_1hop_ensembles(graph,
            #                    drop_version=-1,
            #                    silent=silent,
            #                    config=config)
            # check_models_exists(config, epoch=t_epoch)
        return

    if config.getboolean('Training', 'train_basic_edges'):
        # 1. Build graph + Train 1Hop
        graph = build_space_graph(config, silent=silent, valid_shuffle=True)
        if start_epoch > 0:
            load_2Dtasks(graph, epoch=start_epoch)
        train_2Dtasks(graph,
                      start_epoch=start_epoch,
                      n_epochs=n_epochs,
                      silent=silent,
                      config=config)
        eval_1hop_ensembles(graph,
                            drop_version=-1,
                            silent=silent,
                            config=config)
        return

    if config.getboolean('Training2Iters', 'train_2_iters'):
        use_rgb_to_tsk = config.getboolean('Ensemble', 'use_rgb_to_tsk')
        all_experts = Experts(full_experts=False,
                              use_rgb_to_tsk=use_rgb_to_tsk)
        prepare_store_folders(config=config,
                              iter_no=2,
                              all_experts=all_experts)

        # 00. Build graph
        graph = build_space_graph(config,
                                  silent=silent,
                                  valid_shuffle=False,
                                  iter_no=1)

        load_2Dtasks(graph, epoch=start_epoch)

        # ; 1. Run eval on trainingset2 + save outputs
        save_1hop_ensembles(graph, config=config, iter_no=1)
        '''
        # ; 2. Train on trainset2 using previously saved outputs
        # 00. Build graph
        graph = build_space_graph(config,
                                  silent=silent,
                                  valid_shuffle=False,
                                  iter_no=2)

        load_2Dtasks(graph, epoch=start_epoch)
        train_2Dtasks(
            graph,
            start_epoch=start_epoch,
            n_epochs=0,  # doar pt numaratoare
            silent=silent,
            config=config)

        # ; 3. Run eval on testset
        eval_1hop_ensembles(graph,
                            drop_version=-1,
                            silent=silent,
                            config=config)
        '''
        return

    # 2. Build graph
    graph = build_space_graph(config, silent=silent, valid_shuffle=False)

    # 3. Load 1Hop edges
    load_2Dtasks(graph, epoch=start_epoch)

    # # Per pixel histograms, inside an ensemble
    # plot_per_pixel_ensembles(graph, silent=silent, config=config)
    # drop_version passed as -1 -> no drop
    print("Eval 1Hop ensembles before drop")
    eval_1hop_ensembles(graph, drop_version=-1, silent=silent, config=config)

    # # 3. Drop ill-posed connections
    # drop_version = config.getint('Training', 'drop_version')
    # drop_connections(graph, drop_version)

    # # 4. Eval 1Hop
    # print("Eval 1Hop ensembles after drop (version %i)" % drop_version)
    # drop_version = config.getint('Training', 'drop_version')
    # eval_1hop_ensembles(graph, drop_version, silent=silent, config=config)

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
