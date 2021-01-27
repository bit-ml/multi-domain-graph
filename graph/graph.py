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
        rnd_sampler = torch.Generator()
        self.edges = []
        for i_idx, expert_i in enumerate(all_experts.methods):
            for expert_j in all_experts.methods:
                if expert_i != expert_j:
                    train_only_for_new_expert_b = config.getboolean(
                        'Training', 'train_only_for_new_expert')
                    train_only_for_new_expert = config.get(
                        'Training', 'train_only_for_new_expert')
                    if train_only_for_new_expert_b and train_only_for_new_expert not in [
                            expert_i.identifier, expert_j.identifier
                    ]:
                        continue

                    # # # TO REMOVE
                    # if expert_j.identifier not in [
                    #         # "depth_xtc",
                    #         # "edges_dexined",
                    #         "normals_xtc",
                    #         # "rgb"
                    # ]:
                    #     continue

                    if config.getboolean(
                            'Ensemble', 'restr_dst_domain') and not config.get(
                                'Ensemble',
                                'dst_domain_restr') == expert_j.domain_name:
                        continue
                    new_edge = Edge(config,
                                    expert_i,
                                    expert_j,
                                    device,
                                    rnd_sampler,
                                    silent,
                                    valid_shuffle,
                                    iter_no=iter_no)
                    self.edges.append(new_edge)
                    print("Add edge", str(new_edge))
            # # TO REMOVE
            # break
