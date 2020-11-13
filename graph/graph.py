import torch

from graph.edges.graph_edges import Edge


class MultiDomainGraph:
    def __init__(self,
                 config,
                 experts,
                 device,
                 silent=False,
                 valid_shuffle=True):
        super(MultiDomainGraph, self).__init__()
        self.experts = experts
        self.init_nets(experts, device, silent, config, valid_shuffle)

    def init_nets(self, all_experts, device, silent, config, valid_shuffle):
        rnd_sampler = torch.Generator()
        self.edges = []
        for i_idx, expert_i in enumerate(all_experts.methods):
            for expert_j in all_experts.methods:
                if expert_i != expert_j:
                    new_edge = Edge(config, expert_i, expert_j, device,
                                    rnd_sampler, silent, valid_shuffle)
                    self.edges.append(new_edge)
                    # print("Add edge", str(new_edge))
