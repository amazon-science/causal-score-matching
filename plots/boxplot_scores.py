# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

from typing import Any

import networkx as nx
import numpy as np
from dodiscover.toporder._base import SteinMixin
from matplotlib import pyplot as plt
from numpy._typing import NDArray

from scamuv.data.generate_data import get_confounded_datasets


def is_unconfounded_leaf(graph: nx.DiGraph, node: Any) -> bool:
    if len(list(graph.successors(node))) > 0:
        return False
    return True


def plot_hist_confounders(num_confounders: int):
    datasets = get_confounded_datasets(300, 10, num_confounders, 500)
    num_confounded_pairs = []
    for _, graph in datasets:
        curr_num_conf_pairs = 0
        for x, y in [(x, y) for x in graph.nodes for y in graph.nodes if x != y]:
            if graph.has_edge(x, y) and graph.has_edge(y, x):
                curr_num_conf_pairs += 1
        num_confounded_pairs.append(curr_num_conf_pairs)

    print("Minimum number of confounded pairs: ", np.min(num_confounded_pairs))
    plt.title('Num confounders. {}% unconfounded.'.format(100 * np.mean(np.array(num_confounded_pairs) == 0)))
    plt.hist(num_confounded_pairs, bins=10)
    plt.show()


def plot_boxplots_score(num_confounders: int):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=True)
    axs[0].set_title("Causally sufficient")
    axs[0].boxplot(get_score_samples(0))

    axs[1].set_title("Confounded")
    axs[1].boxplot(get_score_samples(num_confounders))

    for ax in axs:
        ax.yaxis.grid(True)
        ax.set_xticks([y + 1 for y in range(2)],
                      labels=["Leaf", "Non-Leaf"]
                      )
        ax.set_ylabel(r'Score values $\frac{\partial^2}{\partial x^2} \log p(\mathbf{x})$')
        ax.set_yscale('log')
    plt.legend()
    plt.show()


def get_score_samples(num_confounders: int, eta_g: float = 0.001, eta_h: float = 0.001) -> NDArray:
    datasets = get_confounded_datasets(300, 10, num_confounders, 500)
    leaf_scores = []
    other_scores = []
    for i, (data, ground_truth) in enumerate(datasets):
        if i % 10 == 0:
            print("Generated {} datasets".format(i))
        stein_estimator = SteinMixin()
        H_diag = stein_estimator.hessian_diagonal(data.to_numpy(), eta_g, eta_h)
        H_var = H_diag.var(axis=0)
        for i in range(H_var.shape[0]):
            if is_unconfounded_leaf(ground_truth, data.keys()[i]):
                leaf_scores.append(H_var[i])
            else:
                other_scores.append(H_var[i])

    # Have more samples in 'other_scores'. Prune, as boxplot expects same length.
    min_length = min(len(leaf_scores), len(other_scores))
    return np.stack([leaf_scores[:min_length], other_scores[:min_length]], axis=1)


if __name__ == '__main__':
    plot_hist_confounders(5)
    plt.clf()

    plot_boxplots_score(5)
