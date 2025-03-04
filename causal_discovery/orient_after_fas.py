# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

from typing import Any, List

import networkx as nx
import pandas as pd
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.FAS import fas
from causallearn.utils.cit import KCI

from causal_discovery.adascore import AdaScore


class OAFAS(AdaScore):

    def __init__(self,
                 alpha_orientations: float = 0.01,
                 alpha_confounded_leaf: float = 0.01,
                 alpha_separations: float = 0.01,
                 alpha_campruning: float = 0.001,
                 regression: str = 'gam',
                 eta_g: float = 0.001,
                 eta_h: float = 0.001,
                 var_eps: float = 1e-5,
                 cv: int = 3,
                 verbose: bool = False):
        super().__init__(alpha_orientations, alpha_confounded_leaf, alpha_separations, alpha_campruning,
                         regression=regression, eta_g=eta_g, eta_h=eta_h, var_eps=var_eps, cv=cv,
                         verbose=verbose
                         )
        self.verbose = verbose
        self.boundaries = None
        self.visited_nodes = set([])

    def fit(self, data: pd.DataFrame) -> nx.DiGraph:
        self.data = data
        self.boundaries = {node: set([]) for node in data.keys()}
        cit = KCI(data.to_numpy())
        result_skel = fas(data.to_numpy(),
                          [GraphNode(n) for n in data.keys()],
                          alpha=self.alpha_separations,
                          independence_test_method=cit,
                          verbose=self.verbose
                          )[0]
        for edge in result_skel.get_graph_edges():
            self.boundaries[edge.get_node1().get_name()].add(edge.get_node2().get_name())
            self.boundaries[edge.get_node2().get_name()].add(edge.get_node1().get_name())

        self.result_graph = nx.DiGraph()
        self.result_graph.add_nodes_from(data.keys())
        for node in data.keys():
            self._orient_all_incident_edges(node)

        return self.result_graph

    def are_connected(self, first_node: Any, second_node: Any, current_nodes: List[Any]) -> bool:
        # By always returning True we 'disable' the pruning step in _orient_all_incident_edges(), which is superfluous
        # if we use FAS first
        return True
