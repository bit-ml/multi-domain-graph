from graph.edges.graph_edges import Edge


class MultiDomainGraph:
    def __init__(self, experts, device):
        super(MultiDomainGraph, self).__init__()
        self.experts = experts
        self.init_nets(experts, device)

    def init_nets(self, all_experts, device):
        self.edges = []
        for expert_i in all_experts.methods:
            for expert_j in all_experts.methods:
                if expert_i != expert_j:
                    new_edge = Edge(expert_i, expert_j, device)
                    self.edges.append(new_edge)
