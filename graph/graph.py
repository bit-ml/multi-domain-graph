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

    def init_nets(self, all_experts, device, silent, config, valid_shuffle,
                  iter_no):

        restricted_graph_type = config.getint('GraphStructure',
                                              'restricted_graph_type')
        restricted_graph_exp_identifier = config.get(
            'GraphStructure', 'restricted_graph_exp_identifier')

        rnd_sampler = torch.Generator()
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

                    if expert_j.domain_name in ["normals", "rgb"]:
                        # because it has 3 channels
                        bs_test = 20
                        bs_train = 90
                    else:
                        bs_test = 100
                        bs_train = 100
                    new_edge = Edge(config,
                                    expert_i,
                                    expert_j,
                                    device,
                                    rnd_sampler,
                                    silent,
                                    valid_shuffle,
                                    iter_no=iter_no,
                                    bs_train=bs_train,
                                    bs_test=bs_test)
                    self.edges.append(new_edge)
                    print("Add edge", str(new_edge))
