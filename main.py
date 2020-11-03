import torch

from experts.experts import Experts
from graph.edges.graph_edges import (generate_experts_output,
                                     generate_experts_output_with_time)
from graph.graph import MultiDomainGraph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_space_graph():
    all_experts = Experts(full_experts=False)
    md_graph = MultiDomainGraph(all_experts, device)
    return md_graph


def drop_connections(space_graph):
    # for each domain,
    for expert_idx, expert in enumerate(space_graph.experts.all_methods):
        pass


def train_2D_tasks(space_graph):
    # fill spatial nets
    for net_idx, net in enumerate(space_graph.edges):
        print("[%2d] Train" % net_idx, net)
        net.train(epochs=20, device=device)
        # break


def main():
    # from experts.rgb_expert import RGBModel
    # from experts.tracking1_expert import Tracking1Model

    # all_experts_gen = Experts(full_experts=True)
    # generate_experts_output(all_experts_gen.methods)
    # # # generate_experts_output([RGBModel(full_expert=True)])
    # generate_experts_output_with_time([Tracking1Model(full_expert=True)])

    graph = build_space_graph()
    for i in range(7):
        print(("Train all. Epoch:", i))
        train_2D_tasks(graph)

        # drop_connections(graph)


if __name__ == "__main__":
    main()
