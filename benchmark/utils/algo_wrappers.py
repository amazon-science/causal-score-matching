# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

from abc import abstractmethod, ABCMeta
from typing import List

import lingam
import networkx as nx
import numpy as np
import pandas as pd
from causallearn.graph.Edge import Edge
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.FCMBased.lingam import RCD, DirectLiNGAM
from causallearn.search.FCMBased.lingam.CAMUV import execute
from dodiscover import make_context
from dodiscover.toporder import NoGAM
from xgboost import XGBRegressor

from benchmark.data.generate_data import get_pag_skel_with_ada_orientations
from benchmark.data.generator import DataGenerator
from causal_discovery import modified_fci
from benchmark.utils.causal_graphs import MixedGraph, CausalGraph, PAG
from causal_discovery.oracle_adascore import OracleAdaScore
from causal_discovery.orient_after_fas import OAFAS
from causal_discovery.adascore import AdaScore
from causal_discovery.score_independence import ScoreIndependence


class DiscoveryAlgorithm(metaclass=ABCMeta):
    """
    Base class of causal discovery algorithms.
    """

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> CausalGraph:
        raise NotImplementedError()


class RESIT(DiscoveryAlgorithm):
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        # TODO also make regressor parameter
        self.resit = lingam.RESIT(alpha=alpha, regressor=XGBRegressor())

    def fit(self, data: pd.DataFrame) -> MixedGraph:
        self.resit.fit(data)
        g_hat = nx.from_numpy_array(self.resit.adjacency_matrix_.T, create_using=nx.DiGraph)
        nx.relabel_nodes(g_hat, {i: name for (i, name) in enumerate(data.keys())}, copy=False)
        return MixedGraph(g_hat)


class RCDLINGAM(DiscoveryAlgorithm):
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.lingam = RCD(cor_alpha=self.alpha, ind_alpha=self.alpha, shapiro_alpha=self.alpha)

    def fit(self, data: pd.DataFrame) -> MixedGraph:
        self.lingam.fit(data)
        g_hat = nx.from_numpy_array(self.lingam.adjacency_matrix_.T, create_using=nx.DiGraph)
        print(data.keys())
        nx.relabel_nodes(g_hat, {i: name for (i, name) in enumerate(data.keys())}, copy=False)
        return MixedGraph(g_hat)


class LINGAM(DiscoveryAlgorithm):
    def fit(self, data: pd.DataFrame) -> MixedGraph:
        lingam = DirectLiNGAM(random_state=np.random.randint(0, 1000))
        lingam.fit(data)
        g_hat = nx.from_numpy_array(lingam.adjacency_matrix_.T, create_using=nx.DiGraph)
        nx.relabel_nodes(g_hat, {i: name for (i, name) in enumerate(data.keys())}, copy=False)
        return MixedGraph(g_hat)


class FCI(DiscoveryAlgorithm):
    def __init__(self, alpha: float = 0.01, indep_test: str = 'fisherz'):
        self.alpha = alpha
        self.indep_test = indep_test

    def fit(self, data: pd.DataFrame) -> PAG:
        result = fci(data.to_numpy(), alpha=self.alpha, independence_test_method=self.indep_test)[0]
        name_map = {'X{}'.format(i + 1): name for (i, name) in enumerate(data.keys())}
        nodes = [GraphNode(name) for name in data.keys()]
        pag = GeneralGraph(nodes)
        for edge in result.get_graph_edges():
            e_one = pag.get_node(name_map[edge.get_node1().get_name()])
            e_two = pag.get_node(name_map[edge.get_node2().get_name()])
            new_edge = Edge(e_one, e_two, edge.get_endpoint1(), edge.get_endpoint2())
            pag.add_edge(new_edge)
        return PAG(pag)


class ScoreFCI(DiscoveryAlgorithm):
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def fit(self, data: pd.DataFrame) -> PAG:
        cit = ScoreIndependence(data.to_numpy())
        result = modified_fci.fci(data.to_numpy(), alpha=self.alpha, independence_test_method=cit)[0]
        name_map = {'X{}'.format(i + 1): name for (i, name) in enumerate(data.keys())}
        nodes = [GraphNode(name) for name in data.keys()]
        pag = GeneralGraph(nodes)
        for edge in result.get_graph_edges():
            e_one = pag.get_node(name_map[edge.get_node1().get_name()])
            e_two = pag.get_node(name_map[edge.get_node2().get_name()])
            new_edge = Edge(e_one, e_two, edge.get_endpoint1(), edge.get_endpoint2())
            pag.add_edge(new_edge)
        return PAG(pag)


class CAMUV(DiscoveryAlgorithm):
    def __init__(self, alpha: float = 0.01, max_num_parents: int = -1):
        self.alpha = alpha
        self.max_num_parents = max_num_parents

    def fit(self, data: pd.DataFrame) -> MixedGraph:
        if self.max_num_parents == -1:
            self.max_num_parents = len(data.keys())
        node_names = data.keys()
        parents_set, confounded_nodes = execute(data.to_numpy(), self.alpha, num_explanatory_vals=self.max_num_parents)
        graph = nx.DiGraph()
        graph.add_nodes_from(data.keys())
        for i, pa_i in enumerate(parents_set):
            for pa in pa_i:
                graph.add_edge(node_names[pa], node_names[i])
        for first_node, second_node in confounded_nodes:
            graph.add_edge(node_names[first_node], node_names[second_node])
            graph.add_edge(node_names[second_node], node_names[first_node])
        return MixedGraph(graph)


class NoGAMWrapper(DiscoveryAlgorithm):

    def __init__(self, cv: int = 3, alpha: float = 0.01):
        self.alpha = alpha
        self.algo = NoGAM(n_crossval=cv, alpha=self.alpha)

    def fit(self, data: pd.DataFrame) -> MixedGraph:
        context = make_context().variables(data=data).build()
        self.algo.learn_graph(data, context)
        return MixedGraph(self.algo.graph_)


class AdaScoreWrapper(DiscoveryAlgorithm):

    def __init__(self,
                 alpha_orientations: float = 0.01,
                 alpha_confounded_leaf: float = 0.01,
                 alpha_separations: float = 0.01,
                 alpha_campruning: float = 0.001,
                 regression: str = 'gam',
                 cv: int = 3,
                 n_jobs: int = 1):
        self.alpha_orientations = alpha_orientations
        self.alpha_confounded_leaf = alpha_confounded_leaf
        self.alpha_separations = alpha_separations
        self.alpha_campruning = alpha_campruning
        self.regression = regression
        self.algo = AdaScore(alpha_orientation=self.alpha_orientations,
                             alpha_confounded_leaf=self.alpha_confounded_leaf,
                             alpha_separations=self.alpha_separations,
                             alpha_pruning=self.alpha_campruning,
                             regression=self.regression,
                             cv=cv,
                             n_jobs=n_jobs
                             )

    def fit(self, data: pd.DataFrame) -> MixedGraph:
        # We could also directly add this in the AdaScpre class. But it felt cleaner to have the algorithm return the
        # plain nx.DiGraph, so it can be used without this helper MixedGraph class
        result = self.algo.fit(data)
        return MixedGraph(result)


class OAFASWrapper(DiscoveryAlgorithm):

    def __init__(self, alpha_orientations: float = 0.01,
                 alpha_confounded_leaf: float = 0.01,
                 alpha_separations: float = 0.01,
                 alpha_campruning: float = 0.001,
                 cv: int = 3,
                 regression: str = 'gam'):
        self.regression = regression
        self.algo = OAFAS(alpha_orientations, alpha_confounded_leaf, alpha_separations, alpha_campruning,
                          regression=self.regression, cv=cv
                          )

    def fit(self, data: pd.DataFrame) -> MixedGraph:
        # We could also directly add this in the OAFAS class. But it felt cleaner to have the algorithm return the
        # plain nx.DiGraph, so it can be used without this helper MixedGraph class
        result = self.algo.fit(data)
        return MixedGraph(result)

class OracleWrapper(DiscoveryAlgorithm):

    def __init__(self, ground_truth: nx.DiGraph):
        self.algo = OracleAdaScore(ground_truth)

    def fit(self, data: pd.DataFrame) -> MixedGraph:
        result = self.algo.fit(data)
        return MixedGraph(result)


class RandomMixedGraph(DiscoveryAlgorithm):

    def __init__(self, num_hidden: int = 1, erdos_p: float = .3, names: List[str] = None):
        self.num_hidden = num_hidden
        self.erdos_p = erdos_p
        self.names = names

    def fit(self, data: pd.DataFrame) -> MixedGraph:
        num_nodes = len(data.keys())
        g = DataGenerator(num_observed_nodes=num_nodes, num_hidden=self.num_hidden, mechanism='cam', graph_type='dag',
                          erdos_p=self.erdos_p
                          )
        _, ground_truth = g.generate(num_samples=3, noise='gaussian', var=1)
        if self.names is not None:
            nx.relabel_nodes(ground_truth, {old: new for old, new in zip(ground_truth.nodes, self.names)}, copy=False)
        return MixedGraph(get_pag_skel_with_ada_orientations(ground_truth, data.keys()))


class FullyRandomMixedGraph(DiscoveryAlgorithm):

    def fit(self, data: pd.DataFrame) -> MixedGraph:
        order = np.random.permutation(data.keys())
        graph = nx.DiGraph()
        graph.add_nodes_from(order)
        for i, node in enumerate(order):
            for j, second_node in enumerate(order):
                if i < j:
                    r = np.random.rand()
                    if r < 1 / 3.:
                        graph.add_edge(node, second_node)
                    elif r < 2 / 3.:
                        graph.add_edge(node, second_node)
                        graph.add_edge(second_node, node)
        return MixedGraph(graph)


class RandomPAG(DiscoveryAlgorithm):

    def __init__(self, num_hidden: int = 1, erdos_p: float = .3):
        self.num_hidden = num_hidden
        self.erdos_p = erdos_p

    def fit(self, data: pd.DataFrame) -> PAG:
        num_nodes = len(data.keys())
        g = DataGenerator(num_observed_nodes=num_nodes, num_hidden=self.num_hidden, mechanism='cam', graph_type='dag',
                          erdos_p=self.erdos_p
                          )
        _, ground_truth = g.generate(num_samples=3, noise='gaussian', var=1)
        return PAG(PAG.dag_to_pag(ground_truth, list(data.keys())))


def get_algo_from_name(algo_name: str, params: dict) -> DiscoveryAlgorithm:
    if algo_name == 'gam':
        algo = AdaScoreWrapper(alpha_orientations=params['alpha_orientations'],
                               alpha_confounded_leaf=params['alpha_confounded_leaf'],
                               alpha_separations=params['alpha_separations'],
                               alpha_campruning=params['alpha_cam'],
                               cv=params["cv"],
                               n_jobs=params['n_jobs_ada'],
                               regression='gam',
                               )
    elif algo_name == 'ridge':
        algo = AdaScoreWrapper(alpha_orientations=params['alpha_orientations'],
                               alpha_confounded_leaf=params['alpha_confounded_leaf'],
                               alpha_separations=params['alpha_separations'],
                               alpha_campruning=params['alpha_cam'],
                               regression='kernel_ridge',
                               cv=params["cv"],
                               n_jobs=params['n_jobs_ada']
                               )
    elif algo_name == 'falkon':
        algo = AdaScoreWrapper(alpha_orientations=params['alpha_orientations'],
                               alpha_confounded_leaf=params['alpha_confounded_leaf'],
                               alpha_separations=params['alpha_separations'],
                               alpha_campruning=params['alpha_cam'],
                               regression='falkon',
                               cv=params["cv"],
                               n_jobs=params['n_jobs_ada']
                               )
    elif algo_name == 'xgboost':
        algo = AdaScoreWrapper(alpha_orientations=params['alpha_orientations'],
                               alpha_confounded_leaf=params['alpha_confounded_leaf'],
                               alpha_separations=params['alpha_separations'],
                               alpha_campruning=params['alpha_cam'],
                               regression='xgboost',
                               cv=params["cv"],
                               n_jobs=params['n_jobs_ada']
                               )
    elif algo_name == 'linear':
        algo = AdaScoreWrapper(alpha_orientations=params['alpha_orientations'],
                               alpha_confounded_leaf=params['alpha_confounded_leaf'],
                               alpha_separations=params['alpha_separations'],
                               alpha_campruning=params['alpha_cam'],
                               regression='linear',
                               cv=params["cv"],
                               n_jobs=params['n_jobs_ada']
                               )
    elif algo_name == 'oafas':
        algo = OAFASWrapper(alpha_orientations=params['alpha_orientations'],
                            alpha_confounded_leaf=params['alpha_confounded_leaf'],
                            alpha_separations=params['alpha_separations'],
                            alpha_campruning=params['alpha_cam'],
                            cv=params["cv"], regression='kernel_ridge'
                            )
    elif algo_name == 'camuv':
        algo = CAMUV(alpha=params['alpha_others'])
    elif algo_name == 'rcd':
        algo = RCDLINGAM(alpha=params['alpha_others'])
    elif algo_name == 'nogam':
        algo = NoGAMWrapper(cv=params["cv"], alpha=params['alpha_others'])
    elif algo_name == 'lingam':
        algo = LINGAM()
    elif algo_name == 'fci':
        algo = FCI(alpha=params['alpha_others'], indep_test='kci')
    elif algo_name == 'score_fci':
        algo = ScoreFCI(alpha=params['alpha_separations'])
    elif algo_name == 'resit':
        algo = RESIT(alpha=params['alpha_others'])
    elif algo_name == 'random':
        algo = RandomMixedGraph(num_hidden=params['num_hidden'], erdos_p=params['p_edge'])
    elif algo_name == 'random_pag':
        algo = RandomPAG(num_hidden=params['num_hidden'], erdos_p=params['p_edge'])
    elif algo_name == 'fully_random':
        algo = FullyRandomMixedGraph()
    elif algo_name == 'oracle':
        algo = OracleWrapper(params['ground_truth'])
    else:
        raise NotImplementedError(algo_name)
    return algo