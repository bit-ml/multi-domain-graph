import numpy as np
import torch

from graph.edges.graph_edges import Edge


class MultiDomainGraph:
    def __init__(self,
                 config,
                 experts,
                 device,
                 iter_no,
                 silent=False,
                 valid_shuffle=True):
        super(MultiDomainGraph, self).__init__()
        self.experts = experts
        self.init_nets(experts, device, silent, config, valid_shuffle, iter_no)
        print("==================")

    def init_nets(self, all_experts, device, silent, config, valid_shuffle,
                  iter_no):

        restricted_graph_type = config.getint('GraphStructure',
                                              'restricted_graph_type')
        restricted_graph_exp_identifier = config.get(
            'GraphStructure', 'restricted_graph_exp_identifier')

        self.edges = []
        for i_idx, expert_i in enumerate(all_experts.methods):
            for expert_j in all_experts.methods:
                # print("identifiers", expert_i.identifier, expert_j.identifier)
                if expert_i != expert_j:
                    if restricted_graph_type > 0:
                        if restricted_graph_type == 1 and (
                                not expert_i.identifier
                                == restricted_graph_exp_identifier):
                            continue
                        if restricted_graph_type == 2 and (
                                not expert_j.identifier
                                == restricted_graph_exp_identifier):
                            continue
                        if restricted_graph_type == 3 and (
                                not (expert_i.identifier
                                     == restricted_graph_exp_identifier
                                     or expert_j.identifier
                                     == restricted_graph_exp_identifier)):
                            continue

                    model_type = np.int32(
                        config.get('Edge Models', 'model_type'))
                    if model_type == 0:
                        bs_test = 80
                        bs_train = 70
                    else:
                        bs_test = 60
                        bs_train = 40
                    # if expert_j.identifier in ["sem_seg_hrnet"]:
                    #     bs_train = 90

                    print("Add edge [%15s To: %15s]" %
                          (expert_i.identifier, expert_j.identifier),
                          end=' ')
                    new_edge = Edge(config,
                                    expert_i,
                                    expert_j,
                                    device,
                                    silent,
                                    valid_shuffle,
                                    iter_no=iter_no,
                                    bs_train=bs_train,
                                    bs_test=bs_test)
                    self.edges.append(new_edge)
