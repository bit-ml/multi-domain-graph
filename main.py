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
    if config.has_option('Experts', 'selector_map'):
        selector_map_str = config.get('Experts', 'selector_map').split(",")
        selector_map = [int(token) for token in selector_map_str]
    else:
        selector_map = None

    all_experts = Experts(dataset_name=config.get('General', 'DATASET_NAME'),
                          full_experts=False,
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


############################## 1HOP ###############################
def eval_1hop_ensembles(space_graph, silent, config):
    if silent:
        writer = DummySummaryWriter()
    else:
        tb_dir = config.get('Logs', 'tensorboard_dir')
        tb_prefix = config.get('Logs', 'tensorboard_prefix')
        datetime = config.get('Run id', 'datetime')
        writer = SummaryWriter(log_dir=f'%s/%s_1hop_ens_%s' %
                               (tb_dir, tb_prefix, datetime),
                               flush_secs=30)
    for expert in space_graph.experts.methods:
        end_id = expert.identifier

        edges_1hop = []

        # 1. Select edges that ends in end_id
        for edge_xk in space_graph.edges:
            if not edge_xk.trained:
                continue
            if edge_xk.expert2.identifier == end_id:
                edges_1hop.append(edge_xk)

        if len(edges_1hop) == 0:
            continue

        # 2. Eval each ensemble
        edges_order = Edge.eval_all_1hop_ensembles(edges_1hop, device, writer,
                                                   config)

        # 3. print expert indexes - in ascending order of l1 per test set
        ordered_identifiers = []
        for i in edges_order:
            ordered_identifiers.append(edges_1hop[i].expert1.identifier)

        if config.has_option('Ensemble', 'eval_top_edges_nr'):
            eval_top_edges_nr = np.int32(
                config.get('Ensemble', 'eval_top_edges_nr').split(','))

            for top_nr in eval_top_edges_nr:
                to_keep_src_identifiers = ordered_identifiers[0:top_nr]
                print('Top %d sources: ' % (top_nr))
                print(to_keep_src_identifiers)
                if silent:
                    top_writer = DummySummaryWriter()
                else:
                    tb_dir = config.get('Logs', 'tensorboard_dir')
                    tb_prefix = config.get('Logs', 'tensorboard_prefix')
                    datetime = config.get('Run id', 'datetime')
                    top_writer = SummaryWriter(
                        log_dir=f'%s/%s_1hop_ens_top_%d_%s' %
                        (tb_dir, tb_prefix, top_nr, datetime),
                        flush_secs=30)

                edges_1hop = []

                # 1. Select edges that ends in end_id
                for edge_xk in space_graph.edges:
                    if not edge_xk.trained:
                        continue
                    if not edge_xk.expert1.identifier in to_keep_src_identifiers:
                        continue
                    if edge_xk.expert2.identifier == end_id:
                        edges_1hop.append(edge_xk)

                if len(edges_1hop) == 0:
                    continue

                _ = Edge.eval_all_1hop_ensembles(edges_1hop, device,
                                                 top_writer, config)

                top_writer.close()

    writer.close()


def save_1hop_ensembles(space_graph, config, iter_no):
    writer = DummySummaryWriter()

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

    for net_idx, net in enumerate(space_graph.edges):
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
    all_experts = Experts(dataset_name=config.get('General', 'DATASET_NAME'),
                          full_experts=False)
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

    n_iters = config.getint('General', 'n_iters')

    for iteration_idx in np.arange(1, n_iters + 1):

        iter_train_flag = config.getboolean(
            'General', 'Steps_Iter%d_train' % iteration_idx)
        iter_test_flag = config.getboolean('General',
                                           'Steps_Iter%d_test' % iteration_idx)
        iter_saveNextIter_flag = config.getboolean(
            'General', 'Steps_Iter%d_saveNextIter' % iteration_idx)

        # Build graph
        silent = config.getboolean('Logs', 'silent')
        graph = build_space_graph(config,
                                  silent=silent,
                                  valid_shuffle=False,
                                  iter_no=iteration_idx)

        # Load models
        start_epoch = config.getint('Edge Models', 'start_epoch')
        if start_epoch > 0:
            load_2Dtasks(graph, epoch=start_epoch)

        # Train models
        if iter_train_flag:
            n_epochs = config.getint('Edge Models', 'n_epochs')
            train_2Dtasks(graph,
                          start_epoch=start_epoch,
                          n_epochs=n_epochs,
                          silent=silent,
                          config=config)

        # Test models - fixed epoch
        if iter_test_flag:
            eval_1hop_ensembles(graph, silent=silent, config=config)

        # Save data for next iter
        if iter_saveNextIter_flag:
            all_experts = Experts(full_experts=False,
                                  selector_map=config.get(
                                      'Experts', 'selector_map'))
            prepare_store_folders(config=config,
                                  iter_no=iteration_idx + 1,
                                  all_experts=all_experts)

            save_1hop_ensembles(graph, config=config, iter_no=iteration_idx)

        return


if __name__ == "__main__":
    main(sys.argv)
