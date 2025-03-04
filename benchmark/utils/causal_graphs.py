# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

import io
import logging
from typing import Union, Any, List, Dict, Optional

import networkx as nx
import numpy as np
import pydot
from causallearn.graph.Dag import Dag
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.DAG2PAG import dag2pag
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
from cdt.metrics import SHD
from matplotlib import image as mpimg

from benchmark.data.generate_data import get_pag_skel_with_ada_orientations, marginalize
from benchmark.utils.metrics import skeleton_f1, \
    skeleton_precision, \
    skeleton_fpr, \
    skeleton_tpr, \
    direct_edge_precision, \
    direct_edge_recall, direct_edge_f1, bi_edge_f1, bi_edge_recall, bi_edge_precision


class CausalGraph:
    """
    Base class for the causal graphs. The graphs are supposed to be thin wrappers around Networkx graphs or the PAGs
    from Causallearn and provide a unified interface, as well as applicable metrics.
    """

    def __init__(self, graph: Any):
        self.graph = graph

    @staticmethod
    def valid_metrics() -> List[str]:
        return []

    def eval_all_metrics(self, ground_truth: nx.DiGraph, indicate_unobs_direct_path: bool = False) -> Dict[str, float]:
        results = {}
        for metric in self.valid_metrics():
            results[metric] = getattr(self, metric)(ground_truth)
        return results


class PAG(CausalGraph):

    def __init__(self, graph: GeneralGraph):
        super().__init__(graph)

    def skeleton_tpr(self, ground_truth: nx.DiGraph) -> float:
        # skeleton of PAG and MAG are the same.
        # for ease of implementation we convert the PAG to a (networkx) MAG and reuse the skeleton score for
        # networkx DiGraphs
        g_hat_mag = self.draw_random_mag()
        return skeleton_tpr(ground_truth, g_hat_mag)

    def skeleton_fpr(self, ground_truth: nx.DiGraph) -> float:
        # skeleton of PAG and MAG are the same.
        # for ease of implementation we convert the PAG to a (networkx) MAG and reuse the skeleton score for
        # networkx DiGraphs
        g_hat_mag = self.draw_random_mag()
        return skeleton_fpr(ground_truth, g_hat_mag)

    def skeleton_precision(self, ground_truth: nx.DiGraph) -> float:
        # skeleton of PAG and MAG and DAG are the same.
        # for ease of implementation we convert the PAG to a (networkx) MAG and reuse the skeleton score for
        # networkx DiGraphs
        g_hat_mag = self.draw_random_mag()
        return skeleton_precision(ground_truth, g_hat_mag)

    def skeleton_f1(self, ground_truth: nx.DiGraph) -> float:
        # skeleton of PAG and MAG are the same.
        # for ease of implementation we convert the PAG to a (networkx) MAG and reuse the skeleton score for
        # networkx DiGraphs
        g_hat_mag = self.draw_random_mag()
        return skeleton_f1(ground_truth, g_hat_mag)

    def avg_degree(self, _: nx.DiGraph) -> float:
        return float(np.mean([self.graph.get_degree(node) for node in self.graph.get_nodes()]))

    def shd(self, ground_truth: Union[nx.DiGraph, CausalGraph]) -> int:
        if type(ground_truth) == nx.DiGraph:
            latent = list(set(ground_truth.nodes) - {n.get_name() for n in self.graph.get_nodes()})
            ground_truth = self.dag_to_pag(ground_truth, islatent=latent)
        elif type(ground_truth) == PAG:
            ground_truth = ground_truth.graph
        else:
            raise NotImplementedError()
        errors = 0
        for x, y in [(x.get_name(), y.get_name()) for i, x in enumerate(ground_truth.get_nodes()) for j, y in
                     enumerate(ground_truth.get_nodes()) if i < j]:
            if ground_truth.is_adjacent_to(ground_truth.get_node(x), ground_truth.get_node(y)):
                gt_edge = ground_truth.get_edge(ground_truth.get_node(x), ground_truth.get_node(y))
                if not self.graph.is_adjacent_to(self.graph.get_node(x), self.graph.get_node(y)):
                    errors += 1
                    errors += 1 if gt_edge.get_endpoint1() != Endpoint.CIRCLE else 0
                    errors += 1 if gt_edge.get_endpoint2() != Endpoint.CIRCLE else 0
                else:
                    hat_edge = self.graph.get_edge(self.graph.get_node(x), self.graph.get_node(y))
                    if gt_edge.get_endpoint1() != hat_edge.get_endpoint1():
                        errors += 1
                    if gt_edge.get_endpoint2() != hat_edge.get_endpoint2():
                        errors += 1
            else:
                if self.graph.is_adjacent_to(self.graph.get_node(x), self.graph.get_node(y)):
                    hat_edge = self.graph.get_edge(self.graph.get_node(x), self.graph.get_node(y))
                    errors += 1
                    errors += 1 if hat_edge.get_endpoint1() != Endpoint.CIRCLE else 0
                    errors += 1 if hat_edge.get_endpoint2() != Endpoint.CIRCLE else 0
        return errors

    @staticmethod
    def valid_metrics() -> List[str]:
        return ['shd', 'skeleton_f1', 'avg_degree', 'skeleton_tpr', 'skeleton_fpr', 'skeleton_precision']

    @staticmethod
    def dag_to_pag(graph: nx.DiGraph, islatent: Optional[List[Any]] = None) -> GeneralGraph:
        if islatent is None:
            islatent = []
        if not nx.is_directed_acyclic_graph(graph):
            logging.warning('Ground truth is not a DAG!')
            graph = graph.copy()
            while not nx.is_directed_acyclic_graph(
                    graph
            ):  # If due to finite sample effects there is a circle, remove it
                cycle = nx.find_cycle(graph, orientation='original')
                edge = cycle[0]
                graph.remove_edge(edge[0], edge[1])
        nodes_list = [GraphNode(name) for name in graph.nodes]
        dag = Dag(nodes_list)
        for (src, trgt) in graph.edges:
            dag.add_directed_edge(dag.get_node(src), dag.get_node(trgt))
        pag = dag2pag(dag, islatent=[dag.get_node(n) for n in islatent])
        return pag

    def draw_random_mag(self) -> nx.DiGraph:
        aag = nx.DiGraph()
        ccgraph = nx.DiGraph()
        aag.add_nodes_from([n.get_name() for n in self.graph.get_nodes()])
        for edge in self.graph.get_graph_edges():
            if edge.get_endpoint1() == Endpoint.CIRCLE and edge.get_endpoint2() == Endpoint.ARROW:
                aag.add_edge(edge.get_node1().get_name(), edge.get_node2().get_name())  # o-> becomes ->
            elif edge.get_endpoint2() == Endpoint.CIRCLE and edge.get_endpoint1() == Endpoint.ARROW:
                aag.add_edge(edge.get_node2().get_name(), edge.get_node1().get_name())  # <-o becomes <-
            elif edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.ARROW:
                aag.add_edge(edge.get_node1().get_name(), edge.get_node2().get_name())  # keep directed edges
            elif edge.get_endpoint2() == Endpoint.TAIL and edge.get_endpoint1() == Endpoint.ARROW:
                aag.add_edge(edge.get_node2().get_name(), edge.get_node1().get_name())
            elif edge.get_endpoint1() == Endpoint.ARROW and edge.get_endpoint2() == Endpoint.ARROW:
                aag.add_edge(edge.get_node1().get_name(), edge.get_node2().get_name())
                aag.add_edge(edge.get_node2().get_name(), edge.get_node1().get_name())
            else:
                ccgraph.add_edge(edge.get_node1().get_name(), edge.get_node2().get_name())
                ccgraph.add_edge(edge.get_node2().get_name(), edge.get_node1().get_name())

        mag = aag
        while len(ccgraph.nodes) >= 1:
            old_num_nodes = len(ccgraph.nodes)
            # remove "disconnected nodes", i.e. nodes with no undirected edge and no outgoing directed edge left
            ccgraph.remove_nodes_from([n for n in ccgraph.nodes if ccgraph.out_degree(n) == 0])
            remaining_nodes = list(ccgraph.nodes)
            for x in remaining_nodes:
                # if node is sink
                directed_outgoing = [y for y in ccgraph.successors(x) if not ccgraph.has_edge(y, x)]
                if len(directed_outgoing) == 0:
                    # If orientation creates only shielded colliders
                    all_neighbours = {y for y in list(ccgraph.successors(x)) + list(ccgraph.predecessors(x))}
                    bidirected_neighbours = [y for y in ccgraph.successors(x) if ccgraph.has_edge(y, x)]
                    only_shielded = all(
                        [all([ccgraph.has_edge(z, y) or ccgraph.has_edge(y, z) for z in all_neighbours if z != y]) for y
                         in bidirected_neighbours]
                    )
                    if only_shielded:
                        # Orient all bidirected edges towards x in mag
                        for y in bidirected_neighbours:
                            mag.add_edge(y, x)
                        # Remove n from ccgraph
                        ccgraph.remove_node(x)
            if old_num_nodes == len(ccgraph.nodes):
                logging.warning('PAG is no equivalence class of MAGs!')
                # Orient the first x in the remaining graph
                x = remaining_nodes[0]
                for y in [y for y in ccgraph.successors(x) if ccgraph.has_edge(y, x)]:
                    mag.add_edge(y, x)
                # Remove n from ccgraph
                ccgraph.remove_node(x)
        return mag

    def save_graph(self, graph_dir: str):
        with open(graph_dir, 'w') as file:
            file.write(str(self.graph))

    @staticmethod
    def load_graph(graph_dir: str) -> CausalGraph:
        return PAG(txt2generalgraph(graph_dir))

    def visualize(self, ax=None):
        pyd = GraphUtils.to_pydot(self.graph, labels=[n.get_name() for n in self.graph.nodes])
        tmp_png = pyd.create_png(f="png")
        fp = io.BytesIO(tmp_png)
        img = mpimg.imread(fp, format='png')
        ax.set_axis_off()
        ax.imshow(img)


class MixedGraph(CausalGraph):

    def __init__(self, graph: nx.DiGraph):
        super().__init__(graph)

    def direct_edge_precision(self, ground_truth: nx.DiGraph) -> float:
        return direct_edge_precision(ground_truth, self.graph)

    def direct_edge_recall(self, ground_truth: nx.DiGraph) -> float:
        return direct_edge_recall(ground_truth, self.graph)

    def direct_edge_f1(self, ground_truth: nx.DiGraph) -> float:
        return direct_edge_f1(ground_truth, self.graph)

    def bi_edge_precision(self, ground_truth: nx.DiGraph) -> float:
        return bi_edge_precision(ground_truth, self.graph)

    def bi_edge_recall(self, ground_truth: nx.DiGraph) -> float:
        return bi_edge_recall(ground_truth, self.graph)

    def bi_edge_f1(self, ground_truth: nx.DiGraph) -> float:
        return bi_edge_f1(ground_truth, self.graph)

    # def direct_edge_fpr(self, ground_truth: nx.DiGraph) -> float:
    #    return direct_edge_fpr(ground_truth, self.graph)

    def skeleton_tpr(self, ground_truth: nx.DiGraph) -> float:
        return skeleton_tpr(ground_truth, self.graph)

    def skeleton_fpr(self, ground_truth: nx.DiGraph) -> float:
        return skeleton_fpr(ground_truth, self.graph)

    def skeleton_precision(self, ground_truth: nx.DiGraph) -> float:
        return skeleton_precision(ground_truth, self.graph)

    def skeleton_f1(self, ground_truth: nx.DiGraph) -> float:
        return skeleton_f1(ground_truth, self.graph)

    def avg_degree(self, _: nx.DiGraph) -> float:
        return float(np.mean([len(set(self.graph.successors(node)).intersection(set(self.graph.predecessors(node))))
                              for node in self.graph.nodes]
                             )
                     )

    def shd(self, ground_truth: Union[nx.DiGraph, CausalGraph]) -> int:
        if type(ground_truth) == nx.DiGraph:
            return SHD(ground_truth, self.graph, double_for_anticausal=False)
        elif type(ground_truth) == MixedGraph:
            return SHD(ground_truth.graph, self.graph, double_for_anticausal=False)
        else:
            raise NotImplementedError()

    @staticmethod
    def valid_metrics() -> List[str]:
        return ['bi_edge_precision', 'bi_edge_recall', 'bi_edge_f1', 'direct_edge_precision', 'direct_edge_recall',
                'direct_edge_f1', 'shd', 'skeleton_f1', 'skeleton_tpr', 'skeleton_fpr', 'skeleton_precision',
                'avg_degree']

    def eval_all_metrics(self, ground_truth: nx.DiGraph, indicate_unobs_direct_path: bool = False) -> Dict[str, float]:
        results = {}
        marginal_ground_truth = get_pag_skel_with_ada_orientations(ground_truth,
                                                                   list(self.graph.nodes),
                                                                   indicate_unobs_direct_path
                                                                   )
        # Evaluate also w.r.t. to a graph that only contains direct edges from ADMG
        # For PAG output we still calculate it, but it has no meaning
        nobi_ground_truth = marginalize(ground_truth,
                                        list(self.graph.nodes),
                                        indicate_confounding=False,
                                        indicate_unobs_direct_paths=False
                                        )
        for metric in self.valid_metrics():
            results[metric] = getattr(self, metric)(marginal_ground_truth)
            results[metric + '_nobi'] = getattr(self, metric)(nobi_ground_truth)
        return results

    def save_graph(self, graph_dir: str):
        nx.write_graphml(self.graph, graph_dir)

    @staticmethod
    def load_graph(graph_dir: str) -> CausalGraph:
        return MixedGraph(nx.read_graphml(graph_dir))

    def visualize(self, ax=None):
        title = ""
        dpi: float = 200
        fontsize: int = 18

        pydot_g = pydot.Dot(title, graph_type="digraph", fontsize=fontsize)
        pydot_g.obj_dict["attributes"]["dpi"] = dpi

        for node in self.graph.nodes:
            pydot_g.add_node(pydot.Node(node))

        already_bidirected = set([])
        for node_one, node_two in self.graph.edges:
            if self.graph.has_edge(node_two, node_one):
                if (node_one, node_two) not in already_bidirected:
                    dot_edge = pydot.Edge(node_one, node_two, arrowtail='none', arrowhead='none',
                                          style='dotted'
                                          )
                    already_bidirected.add((node_two, node_one))
                    pydot_g.add_edge(dot_edge)
            else:
                dot_edge = pydot.Edge(node_one, node_two, arrowtail='none', arrowhead='normal')
                pydot_g.add_edge(dot_edge)

        tmp_png = pydot_g.create_png(f="png")
        fp = io.BytesIO(tmp_png)
        img = mpimg.imread(fp, format='png')
        ax.set_axis_off()
        ax.imshow(img)
