import torch

from experts.experts import Experts
from graph.graph import MultiDomainGraph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_space_graph():
    all_experts = Experts()
    md_graph = MultiDomainGraph(all_experts, device)
    return md_graph


def train_2D_tasks(space_graph):
    # fill spatial nets

    for net in space_graph.edges:
        net.train(epochs=5, device=device)
        break


def main():
    graph = build_space_graph()
    train_2D_tasks(graph)


if __name__ == "__main__":
    main()
