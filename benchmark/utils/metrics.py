# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

from typing import List, Any, Tuple

import networkx as nx
from sklearn.metrics import precision_score, recall_score, f1_score


def direct_edge_lists(gt: nx.DiGraph, g_hat: nx.DiGraph) -> Tuple[List[bool], List[bool]]:
    true_edges = []
    est_edges = []
    for x, y in [(x, y) for x in gt.nodes for y in gt.nodes if x != y]:
        true_edges.append(gt.has_edge(x, y) and not gt.has_edge(y, x))
        est_edges.append(g_hat.has_edge(x, y) and not g_hat.has_edge(y, x))
    return true_edges, est_edges


def direct_edge_precision(gt: nx.DiGraph, g_hat: nx.DiGraph) -> float:
    true, est = direct_edge_lists(gt, g_hat)
    return precision_score(true, est, zero_division=1)


def direct_edge_recall(gt: nx.DiGraph, g_hat: nx.DiGraph) -> float:
    true, est = direct_edge_lists(gt, g_hat)
    return recall_score(true, est, zero_division=1)


def direct_edge_f1(gt: nx.DiGraph, g_hat: nx.DiGraph) -> float:
    true, est = direct_edge_lists(gt, g_hat)
    return f1_score(true, est, zero_division=1)


def bi_edge_lists(gt: nx.DiGraph, g_hat: nx.DiGraph) -> Tuple[List[bool], List[bool]]:
    true_edges = []
    est_edges = []
    for x, y in [(x, y) for (i, x) in enumerate(gt.nodes) for (j, y) in enumerate(gt.nodes) if i < j]:
        true_edges.append(gt.has_edge(x, y) and gt.has_edge(y, x))
        est_edges.append(g_hat.has_edge(x, y) and g_hat.has_edge(y, x))
    return true_edges, est_edges


def bi_edge_precision(gt: nx.DiGraph, g_hat: nx.DiGraph) -> float:
    true, est = bi_edge_lists(gt, g_hat)
    return precision_score(true, est, zero_division=1)


def bi_edge_recall(gt: nx.DiGraph, g_hat: nx.DiGraph) -> float:
    true, est = bi_edge_lists(gt, g_hat)
    return recall_score(true, est, zero_division=1)


def bi_edge_f1(gt: nx.DiGraph, g_hat: nx.DiGraph) -> float:
    true, est = bi_edge_lists(gt, g_hat)
    return f1_score(true, est, zero_division=1)


def dtop(ground_truth: nx.DiGraph, order: List[Any], ignore_bidirected: bool = False) -> int:
    if ignore_bidirected:
        # Remove bidirectional edges, so that they won't count as an error
        ground_truth = ground_truth.copy()
        for i, first_node in enumerate(order[:-1]):
            for second_node in order[i + 1:]:
                if ground_truth.has_edge(first_node, second_node) and ground_truth.has_edge(second_node, first_node):
                    ground_truth.remove_edge(first_node, second_node)
                    ground_truth.remove_edge(second_node, first_node)

    num_errors = 0
    for i, first_node in enumerate(order[:-1]):
        for j, second_node in enumerate(order[i + 1:]):
            if ground_truth.has_edge(second_node, first_node):
                print(first_node, second_node)
                num_errors += 1
    return num_errors


def skeleton_tpr(gt: nx.DiGraph, g_hat: nx.DiGraph) -> float:
    tp = 0
    p = 0
    for x, y in [(x, y) for x in gt.nodes for y in gt.nodes if x != y]:
        if gt.has_edge(x, y) or gt.has_edge(y, x):
            p += 1
            if g_hat.has_edge(x, y) or g_hat.has_edge(y, x):
                tp += 1
    return float(tp) / p if p != 0 else 1.


def skeleton_fpr(gt: nx.DiGraph, g_hat: nx.DiGraph) -> float:
    fp = 0
    n = 0
    for x, y in [(x, y) for x in gt.nodes for y in gt.nodes if x != y]:
        if not gt.has_edge(x, y) and not gt.has_edge(y, x):
            n += 1
            if g_hat.has_edge(x, y) or g_hat.has_edge(y, x):
                fp += 1
    return float(fp) / n if n != 0 else 1.


def skeleton_precision(gt: nx.DiGraph, g_hat: nx.DiGraph) -> float:
    tp = 0
    fp = 0
    for x, y in [(x, y) for x in gt.nodes for y in gt.nodes if x != y]:
        if g_hat.has_edge(x, y) or g_hat.has_edge(y, x):
            if gt.has_edge(x, y) or gt.has_edge(y, x):
                tp += 1
            else:
                fp += 1
    return float(tp) / (fp + tp) if fp + tp != 0 else 1.


def skeleton_f1(gt: nx.DiGraph, g_hat: nx.DiGraph) -> float:
    ppv = skeleton_precision(gt, g_hat)
    tpr = skeleton_tpr(gt, g_hat)
    return (2 * ppv * tpr / (ppv + tpr)) if (ppv + tpr) != 0 else 0
