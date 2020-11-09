from graph.edges.graph_edges import Edge


class MultiDomainGraph:
    def __init__(self, config, experts, device, silent=False):
        super(MultiDomainGraph, self).__init__()
        self.experts = experts
        self.init_nets(experts, device, silent, config)

    def init_nets(self, all_experts, device, silent, config):
        self.edges = []
        for expert_i in all_experts.methods:
            for expert_j in all_experts.methods:
                if expert_i != expert_j:
                    new_edge = Edge(config, expert_i, expert_j, device, silent)
                    print("Add edge:", new_edge)
                    self.edges.append(new_edge)
