import numpy as np
import torch
from scipy.stats import pearsonr

from experts.experts import Experts
from graph.edges.graph_edges import (generate_experts_output,
                                     generate_experts_output_with_time)
from graph.graph import MultiDomainGraph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_space_graph(silent):
    all_experts = Experts(full_experts=False)
    md_graph = MultiDomainGraph(all_experts, device, silent=silent)
    return md_graph


def ensemble_vs_edge(metrics, edge_idx):
    ensemble_values = metrics.mean(axis=0)
    edge_values = metrics[edge_idx]
    diff_array = ensemble_values - edge_values

    # v1, with std
    std = np.std(diff_array)
    is_outlier = std > 0.01
    print("[Edge_idx %d] std %.2f" % (edge_idx, std))

    # v2, with pearson
    r_corr, p_value = pearsonr(ensemble_values, diff_array)
    is_outlier = abs(r_corr) < 0.5
    print("[Edge_idx %d] r_corr %.2f" % (edge_idx, r_corr))

    return is_outlier


def evaluate_all_edges(ending_edges):
    metrics = []
    for edge in ending_edges:
        edge_l2_loss, edge_l1_loss = edge.eval_detailed(device)
        metrics.append(np.array(edge_l1_loss))

    return np.array(metrics)


def drop_connections(space_graph):
    # for each node domain (= expert in our case)
    for expert_idx, expert in enumerate(space_graph.experts.methods):
        ending_edges = []
        # all edges reaching this node
        for edge in space_graph.edges:
            if edge.expert2.str_id == expert.str_id:
                ending_edges.append(edge)

        pred_metrics_res = evaluate_all_edges(ending_edges)

        # ensembles vs single model
        for edge_idx, edge in enumerate(ending_edges):
            is_outlier = ensemble_vs_edge(pred_metrics_res, edge_idx)
            print("Is edge %s outlier?" % str(edge), is_outlier)
            edge.ill_posed = is_outlier
        break


def train_2hop_2Dtasks(space_graph, epochs):
    # use bool: edge.ill_posed
    '''
    Like train_1hop_2Dtasks, but use ensembles as GT:
    '''
    for net_idx, net in enumerate(space_graph.edges):
        print("[net_idx %2d] Train with 2-hop supervision" % (net_idx), net)
        net.train_from_2hops_ensemble(graph=space_graph,
                                      epochs=epochs,
                                      device=device)


def train_1hop_2Dtasks(space_graph, epochs):
    # fill spatial nets
    for net_idx, net in enumerate(space_graph.edges):
        print("[%2d] Train" % net_idx, net)
        net.train(epochs=epochs, device=device)
        # break


def main():
    from experts.rgb_expert import RGBModel
    from experts.tracking1_expert import Tracking1Model

    # TODO: muta asta
    # all_experts_gen = Experts(full_experts=True)
    # generate_experts_output(all_experts_gen.methods)
    # # generate_experts_output([RGBModel(full_expert=True)])
    # generate_experts_output_with_time([Tracking1Model(full_expert=True)])
    # 1. Build graph
    graph = build_space_graph(silent=False)

    # 2. Train 1hop
    train_1hop_2Dtasks(graph, epochs=60)

    for i in range(1):
        print(("Train 1hop. Epoch:", i))
        train_1hop_2Dtasks(graph, epochs=1)

        # # Drop ill-posed connections
        # drop_connections(graph)

        # 3. Train 2hop
        print(("Train 2hop. Epoch:", i))
        train_2hop_2Dtasks(graph, epochs=3)

        # # Drop ill-posed connections
        # drop_connections(graph)


if __name__ == "__main__":
    main()
