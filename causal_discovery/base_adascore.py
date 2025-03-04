# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import combinations
from typing import Any, List, Collection, Optional, Dict

import networkx as nx
import pandas as pd


def _all_subsets(s: Collection[Any]) -> List[Any]:
    for subset_size in range(len(s) + 1):
        for subset in combinations(s, subset_size):
            yield list(subset)


def _no_direct_edge(g: nx.DiGraph, node: Any, second_node: Any) -> bool:
    no_edge = not g.has_edge(node, second_node) and not g.has_edge(second_node, node)
    bidir = g.has_edge(node, second_node) and g.has_edge(second_node, node)
    return no_edge or bidir


class BaseAdaScore(ABC):

    @abstractmethod
    def get_unconfounded_leaf(self, relevant_nodes: List[Any], current_nodes: List[Any]) -> Optional[Any]:
        raise NotImplementedError()

    @abstractmethod
    def are_connected(self, first_node: Any, second_node: Any, current_nodes: List[Any]) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def orient_edge(self, node, second_node, neighbourhood_node, neighbourhood_second):
        raise NotImplementedError()

    @abstractmethod
    def prune_edge(self, first_node, second_node, neighbourhood) -> bool:
        raise NotImplementedError()

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.order: List[Any] = None
        self.remaining_nodes: List[Any] = None
        self.result_graph = nx.DiGraph()
        self.visited_nodes = set([])
        self.boundaries = {}
        self.data = None

    def fit(self, data: pd.DataFrame) -> nx.DiGraph:
        if self.verbose:
            print('Training: ', type(self))
        self.data = data
        self.order = []
        self.remaining_nodes = list(data.keys())
        self.result_graph = nx.DiGraph()
        self.result_graph.add_nodes_from(self.remaining_nodes)
        for node in data.keys():
            self.boundaries[node] = set(data.keys()) - {node}

        for i in range(len(self.remaining_nodes)):
            self._initial_prune_all_neighbourhoods(self.remaining_nodes)
            current_leaf, is_leaf = self.get_unconfounded_leaf(self.remaining_nodes, self.remaining_nodes)
            if is_leaf:
                # current_leaf can be safely removed
                self._remove_unconfounded_leaf(current_leaf)
            else:
                self._remove_confounded_node(current_leaf)

            if len(self.remaining_nodes) == 1:
                self.order = (self.order + self.remaining_nodes)[::-1]
                self._prune_bidirected_edges()
                return self.result_graph

        raise Exception("Somethings wrong")

    def _remove_unconfounded_leaf(self, node: Any):
        if self.verbose:
            print("Found unconfounded leaf ", node)
        self.order.append(node)
        self.remaining_nodes.remove(node)
        for parent in self.boundaries[node]:
            if self.verbose:
                print("Add ", parent, "->", node)
            self.result_graph.add_edge(parent, node)

    def _remove_confounded_node(self, node: Any, recursion_history: Optional[Dict[Any, int]] = None):
        if recursion_history is None:
            recursion_history = defaultdict(lambda: 0)
        else:
            recursion_history[node] = recursion_history[node] + 1
        if self.verbose:
            print("Explore confounded node ", node)

        self._orient_all_incident_edges(node)

        # If node has direct children, recurse on them until one without is found
        for child in self.boundaries[node]:
            if (self.result_graph.has_edge(node, child)
                    and not self.result_graph.has_edge(child, node)
                    # prevents endless recursion if circle due to finite samples
                    and not recursion_history[child] > len(self.result_graph.nodes)):
                self._remove_confounded_node(child, recursion_history)
                return
        if self.verbose:
            print("Remove confounded leaf ", node)

        # Remove node from remaining nodes
        self.order.append(node)
        self.remaining_nodes.remove(node)

    def _orient_all_incident_edges(self, node: Any):
        if node in self.visited_nodes:
            return
        self.visited_nodes.add(node)

        for second_node in self.boundaries[node].copy():
            if (not self.result_graph.has_edge(node, second_node)
                    and not self.result_graph.has_edge(second_node, node)):
                if self.prune_edge(node, second_node, self.boundaries[node].union({node})):
                    self.boundaries[node].discard(second_node)
                    self.boundaries[second_node].discard(node)

        for second_node in self.boundaries[node]:
            if second_node not in self.visited_nodes:
                orientation = self.orient_edge(node, second_node, self.boundaries[node], self.boundaries[second_node])
                if orientation == '->':
                    if self.verbose:
                        print("Add ", node, "->", second_node)
                    self.result_graph.add_edge(node, second_node)
                elif orientation == '<-':
                    if self.verbose:
                        print("Add ", second_node, "->", node)
                    self.result_graph.add_edge(second_node, node)
                else:
                    if self.verbose:
                        print("Add ", node, "<->", second_node)
                    self.result_graph.add_edge(node, second_node)
                    self.result_graph.add_edge(second_node, node)

    def _initial_prune_all_neighbourhoods(self, current_nodes: List[Any]):
        for i, node in enumerate(current_nodes):
            for j, node_two in enumerate(current_nodes):
                if i < j and node_two in self.boundaries[node].copy():
                    if not self.are_connected(node, node_two, current_nodes):
                        self.boundaries[node].discard(node_two)
                        self.boundaries[node_two].discard(node)
        self.boundaries = {n: self.boundaries[n].intersection(current_nodes) for n in current_nodes}

    def _prune_bidirected_edges(self):
        for i, node in enumerate(self.data.keys()):
            for j, second_node in enumerate(self.data.keys()):
                if self.result_graph.has_edge(node, second_node) and self.result_graph.has_edge(second_node, node):
                    neighbourhood = set(self.result_graph.successors(node)).union(set(self.result_graph.predecessors(
                        node
                    )
                    )
                    ) - {node, second_node}
                    if self.prune_edge(node, second_node, neighbourhood):
                        self.result_graph.remove_edge(node, second_node)
                        self.result_graph.remove_edge(second_node, node)
                        break
