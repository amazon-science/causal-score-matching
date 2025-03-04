# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

from collections import defaultdict
from typing import Iterable, Tuple, Any, Set, List

import networkx as nx
import pandas as pd

from benchmark.data.generator import DataGenerator
from causal_discovery.base_adascore import _all_subsets


def get_confounded_datasets(num_datasets: int,
                            num_nodes: int,
                            num_hidden: int,
                            num_samples: int,
                            erdos_p: int = 2,
                            mechansim: str = 'ram'
                            ) -> List[Tuple[pd.DataFrame, nx.DiGraph]]:
    """
    Generates a list of pairs of synthetic data matrices and matching ground truth graphs
    :param num_datasets: Number of dataset/ground truth pairs generated
    :param num_nodes: Number of observable nodes in the data matrix
    :param num_hidden: Number of hidden nodes that appear in the ground truth graph but not in the data matrix
    :param num_samples: Number of samples in the data matrix of each pair
    :param expected_out_degree: Expected number of outgoing edges per node in the ground truth graphs of each pair
    :return: List of pairs of data and corresponding ground truth graph. (Note that the ground truth graph also contains
    the hidden nodes)
    """
    datasets = []
    for _ in range(num_datasets):
        g = DataGenerator(num_observed_nodes=num_nodes, num_hidden=num_hidden, mechanism=mechansim, graph_type='dag',
                          erdos_p=erdos_p)
        data, ground_truth = g.generate(num_samples=num_samples, noise='uniform', var=1)
        datasets.append((data, ground_truth))

    return datasets


def get_pag_skel_with_ada_orientations(graph: nx.DiGraph,
                                       remaining_nodes: Iterable[Any],
                                       indicate_unobs_direct_path: bool = True) -> nx.DiGraph:
    skeleton = nx.Graph()
    skeleton.add_nodes_from(remaining_nodes)
    for i, first in enumerate(remaining_nodes):
        for j, second in enumerate(remaining_nodes):
            if i < j:
                separable = False
                # TODO move _all_subsets() somewhere else
                for cond_set in _all_subsets(set(remaining_nodes) - {first, second}):
                    if nx.d_separated(graph, {first}, {second}, set(cond_set)):
                        separable = True
                        break
                if not separable:
                    skeleton.add_edge(first, second)

    result = nx.DiGraph()
    result.add_nodes_from(remaining_nodes)
    potential_edge = defaultdict(lambda: defaultdict(lambda: False))
    for first in remaining_nodes:
        for second in remaining_nodes:
            if first != second and skeleton.has_edge(first, second):
                for subset in _all_subsets(list(skeleton.neighbors(second))):
                    if first in subset:
                        marginal_graph = marginalize(graph,
                                                     subset + [second],
                                                     indicate_confounding=True,
                                                     indicate_unobs_direct_paths=indicate_unobs_direct_path
                                                     )
                        if not list(marginal_graph.successors(second)):
                            potential_edge[first][second] = True

    for first in remaining_nodes:
        for second in remaining_nodes:
            if first != second and skeleton.has_edge(first, second):
                if potential_edge[first][second] and not potential_edge[second][first]:
                    result.add_edge(first, second)

    for first in remaining_nodes:
        for second in remaining_nodes:
            if first != second and skeleton.has_edge(first, second):
                if not result.has_edge(first, second) and not result.has_edge(second, first):
                    result.add_edge(first, second)
                    result.add_edge(second, first)
    return result


def marginalize(graph: nx.DiGraph,
                remaining_nodes: Iterable[Any],
                indicate_confounding: bool = True,
                indicate_unobs_direct_paths: bool = False) -> nx.DiGraph:
    """
    Marginalize all nodes except for the 'remaining_nodes' from  the given graph.
    :param graph: The graph from which the nodes are marginalized.
    :param remaining_nodes: The nodes that are not marginalized, i.e. the observable nodes after marginalization.
    :param indicate_confounding: If True (default) include a bidirected edge in the output between two nodes, if they
    have a common ancestor that has been marginalized out.
    :param indicate_unobs_direct_paths: If True (default is False) include a bidirected edge in the output between two
    nodes, if there is a path between them, such that the last (non-endpoint) node has been marginalized out, i.e. a
    UDP as defined by Maeda et al. (2022).
    :return: A nx.DiGraph that represents the causal structure after marginalizing out all nodes execpt for the
    'remaining_nodes'.
    """
    g_marginalised = graph.copy()
    remaining_nodes = set(remaining_nodes)
    confounded_nodes = set([])
    nodes_with_unobs_direct_path = set([])
    for marginalised_node in set(g_marginalised.nodes) - remaining_nodes:
        g_marginalised, confounded_nodes, nodes_with_unobs_direct_path = _marginalise_node(g_marginalised,
                                                                                           marginalised_node,
                                                                                           confounded_nodes,
                                                                                           nodes_with_unobs_direct_path
                                                                                           )

    for x, y in confounded_nodes:
        # Add bidirectional edge for confounded nodes
        if x in remaining_nodes and y in remaining_nodes:
            if indicate_confounding:
                g_marginalised.add_edge(x, y)
                g_marginalised.add_edge(y, x)

    for x, y in nodes_with_unobs_direct_path:
        # Add bidirectional edge for nodes with unobserved direct path
        if x in remaining_nodes and y in remaining_nodes:
            if indicate_unobs_direct_paths:
                g_marginalised.add_edge(x, y)
                g_marginalised.add_edge(y, x)
    return g_marginalised


def _marginalise_node(graph: nx.DiGraph,
                      marginalised_node: Any,
                      confounded_nodes: Set,
                      nodes_with_unobs_direct_path: Set) -> Tuple[nx.DiGraph, Set[Any], Set[Any]]:
    """
    Marginalize the 'marginalized_node' from  the given graph.
    :param graph: The graph from which the nodes are marginalized.
    :param marginalised_node: The nodes that is removed, i.e. the unobservable node after marginalization.
    :param confounded_nodes: Set of nodes that already have a hidden common cause between them.
    :param nodes_with_unobs_direct_path: Set of nodes that already have an unobserved direct path between them.
    :return: Triplet of the marginalized graph, the updated set of confounded nodes and the updated set of nodes with
    unobserved direct path between them.

    Note, that the returned graph contains edges between parents and children of
    'marginalized_node' to indicate that there is a path between them (even if these pairs are also added to the
    'nodes_with_unobs_direct_path').
    """
    for anc in graph.nodes:
        if nx.has_path(graph, anc, marginalised_node):
            for succ in graph.successors(marginalised_node):
                nodes_with_unobs_direct_path.add((anc, succ))  # Edge will be bidirected in CAM-UV
    # Add egdes from all predecessors to sucessors, i.e. replace  x -> node -> y with a new edge x -> y
    # If indicate_unobs_direct_paths is True, this edge will be turned into a bidirected one later
    for pre in graph.predecessors(marginalised_node):
        for succ in graph.successors(marginalised_node):
            if pre != succ:
                graph.add_edge(pre, succ)

    for suc_one in graph.successors(marginalised_node):  # Add nodes to set of confounded nodes, if directly confounded
        for suc_two in graph.successors(marginalised_node):
            if suc_two != suc_one:
                confounded_nodes.add((suc_one, suc_two))

        # Add (x, suc_one) to confounded nodes if z is removed, z -> suc_one and (x, z) are confounded, i.e. if there is
        # 'indirect' confounding.
        for x, _ in [(x, y) for (x, y) in confounded_nodes if y == marginalised_node]:
            if x != suc_one:
                confounded_nodes.add((x, suc_one))

    graph.remove_node(marginalised_node)
    return graph, confounded_nodes, nodes_with_unobs_direct_path
