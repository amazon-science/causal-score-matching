# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

from typing import Any, List, Optional

import networkx as nx

from benchmark.data.generate_data import marginalize
from causal_discovery.base_adascore import BaseAdaScore, _all_subsets


class OracleAdaScore(BaseAdaScore):

    def __init__(self, ground_truth: nx.DiGraph, verbose: bool = False):
        super().__init__(verbose)
        self.ground_truth = ground_truth.copy()

    def get_unconfounded_leaf(self, relevant_nodes: List[Any], current_nodes: List[Any]) -> Optional[Any]:
        marginal_graph = marginalize(self.ground_truth.copy(),
                                     current_nodes,
                                     indicate_confounding=True,
                                     indicate_unobs_direct_paths=True
                                     )
        for node in marginal_graph:
            if not list(marginal_graph.successors(node)):
                return node, True
        return current_nodes[0], False

    def orient_edge(self, node, second_node, neighbourhood_node, neighbourhood_second):
        for subset in _all_subsets(neighbourhood_node - {second_node}):
            marginal_graph = marginalize(self.ground_truth.copy(),
                                         subset + [node, second_node],
                                         indicate_confounding=True,
                                         indicate_unobs_direct_paths=True
                                         )
            if not list(marginal_graph.successors(node)) and marginal_graph.has_edge(second_node, node):
                return '<-'
        for subset in _all_subsets(neighbourhood_second - {node}):
            marginal_graph = marginalize(self.ground_truth.copy(),
                                         subset + [node, second_node],
                                         indicate_confounding=True,
                                         indicate_unobs_direct_paths=True
                                         )
            if marginal_graph.has_edge(node, second_node) and not list(marginal_graph.successors(second_node)):
                return '->'
        return '-'

    def are_connected(self, first_node: Any, second_node: Any, current_nodes: List[Any]) -> bool:
        return not nx.d_separated(self.ground_truth,
                                  {first_node},
                                  {second_node},
                                  set(current_nodes) - {first_node, second_node}
                                  )

    def prune_edge(self, first_node, second_node, neighbourhood) -> bool:
        for subset in _all_subsets(neighbourhood):
            if nx.d_separated(self.ground_truth, {first_node}, {second_node}, subset):
                return True
        return False
