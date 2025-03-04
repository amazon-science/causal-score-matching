# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

from collections import defaultdict
from typing import Any, List, Tuple, Optional

import networkx as nx
import numpy as np
import pandas as pd
from causallearn.utils.cit import CIT
from dodiscover.toporder._base import SteinMixin, CAMPruning
from dodiscover.toporder.utils import kernel_width
from joblib import Parallel, delayed
from numpy._typing import NDArray
from pygam import LinearGAM
from scipy.stats import ttest_1samp
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from causal_discovery.base_adascore import BaseAdaScore, _all_subsets


class AdaScore(BaseAdaScore, SteinMixin, CAMPruning):

    def __init__(self,
                 alpha_confounded_leaf: float,
                 alpha_orientation: float,
                 alpha_separations: float,
                 alpha_ridge: float = 0.01,
                 regression: str = 'kernel_ridge',
                 eta_g: float = 0.001,
                 eta_h: float = 0.001,
                 var_eps: float = 1e-5,
                 cv: int = 5,
                 n_splines: int = 10,
                 splines_degree: int = 3,
                 alpha_pruning: float = 0.001,
                 prune_kci: bool = False,
                 use_cache: bool = True,
                 n_jobs: int = 1,
                 verbose: bool = False):
        super().__init__(verbose=verbose)
        self.alpha_confounded_leaf = alpha_confounded_leaf
        self.alpha_orientation = alpha_orientation
        self.alpha_separations = alpha_separations
        self.alpha_ridge = alpha_ridge
        self.regression = regression
        self.eta_g = eta_g
        self.eta_h = eta_h
        self.score_eps = var_eps
        self.cv = cv
        self.use_cache = use_cache
        self.cache = defaultdict(lambda: None)
        self.ordered_cache = set([])
        self.n_splines = n_splines
        self.degree = splines_degree
        self.alpha = alpha_pruning  # pruning method expects threshold to be called just 'alpha'
        self.prune_kci = prune_kci
        self._residual_oracle = {}
        self.cit = None
        self.cit_idx = None
        self.n_jobs = n_jobs

    def fit(self, data: pd.DataFrame) -> nx.DiGraph:
        self.cit = CIT(data.to_numpy(), 'kci')
        self.cit_idx = {n: i for i, n in enumerate(data.keys())}
        super().fit(data)
        self._prune_directed_edges()
        return self.result_graph

    def _prune_directed_edges(self):
        # Pruning step
        for node in self.result_graph.nodes:
            potential_parents = [p for p in self.result_graph.predecessors(node) if
                                 not self.result_graph.has_edge(node, p)]
            if potential_parents:
                empty_prior_knowledge = nx.DiGraph()
                empty_prior_knowledge.add_nodes_from(self.result_graph.nodes)
                parents = self._variable_selection(self.data[potential_parents].to_numpy(),
                                                   self.data[node].to_numpy(),
                                                   potential_parents,
                                                   node,
                                                   empty_prior_knowledge
                                                   )
                for non_parent in set(potential_parents) - set(parents):
                    self.result_graph.remove_edge(non_parent, node)

    def get_regression(self, num_nodes: int):
        if self.regression == 'kernel_ridge':
            return KernelRidge(kernel='rbf', gamma=0.01, alpha=self.alpha_ridge)
        elif self.regression == 'gam':
            return LinearGAM()
        elif self.regression == 'xgboost':
            return XGBRegressor()
        elif self.regression == 'linear':
            return LinearRegression()
        else:
            raise NotImplementedError(self.regression)

    def _get_delta(self, node: Any, current_nodes: List[Any]) -> Tuple[NDArray, NDArray, NDArray]:
        if len(current_nodes) == 1 and node in current_nodes:
            return 0, self.data[node].to_numpy(), np.expand_dims(np.abs(self.data[node].to_numpy()) ** 2, axis=1)
        current_nodes = list(np.sort(current_nodes))  # Sort for more cache hits
        _, score_vector, _, _ = self._get_score_and_helpers(current_nodes)
        score = score_vector[:, current_nodes.index(node)]
        data = self.data[current_nodes]
        predictors = data.loc[:, data.columns != node].to_numpy()
        if (frozenset(current_nodes), node) in self._residual_oracle:  # Only used for testing
            node_residuals = self._residual_oracle[(frozenset(current_nodes), node)]
        else:
            node_residuals = self._cv_predict(predictors, data[node].to_numpy())
        score_residual = self._cv_predict(np.expand_dims(node_residuals, -1), score)
        return (np.abs(score_residual) ** 2), node_residuals, predictors

    def get_unconfounded_leaf(self, relevant_nodes: List[Any], current_nodes: List[Any]) -> Optional[Any]:
        deltas = Parallel(n_jobs=self.n_jobs)(
            delayed(
                lambda n: self._get_delta(n, list(self.boundaries[n].union({n})))
            )(node) for node in relevant_nodes
        )
        deltas = {n: d for n, d in zip(relevant_nodes, deltas)}
        current_leaf = min(deltas, key=lambda n: np.mean(deltas[n][0]))
        _, node_residual, predictors = deltas[current_leaf]
        cit = CIT(np.concatenate([np.expand_dims(node_residual, -1), predictors], axis=1), method='kci')
        p = cit(0, np.arange(1, predictors.shape[1] + 1))

        return current_leaf, p >= self.alpha_confounded_leaf


    def orient_edge(self, node, second_node, neighbourhood_node, neighbourhood_second):
        with Parallel(n_jobs=self.n_jobs) as parallel:
            first_deltas = parallel(
                delayed(lambda s: self._get_delta(node, s + [node, second_node])
                        )(subset) for subset in _all_subsets(neighbourhood_node - {second_node})
            )
            second_deltas = parallel(
                delayed(lambda s: self._get_delta(second_node, s + [node, second_node])
                        )(subset) for subset in _all_subsets(neighbourhood_second - {node})
            )
        first_min = min(first_deltas, key=lambda n: np.mean(n[0]))
        _, node_residual, predictors = first_min
        if len(predictors.shape) == 1:
            predictors = np.expand_dims(predictors, axis=-1)
        cit_first = CIT(np.concatenate([np.expand_dims(node_residual, -1), predictors], axis=1), method='kci')
        p_first = cit_first(0, np.arange(1, predictors.shape[1] + 1))
        second_min = min(second_deltas, key=lambda n: np.mean(n[0]))
        _, node_residual, predictors = second_min
        if len(predictors.shape) == 1:
            predictors = np.expand_dims(predictors, axis=-1)
        joint_df = np.concatenate([np.expand_dims(node_residual, -1), predictors], axis=1)
        cit_second = CIT(joint_df, method='kci')
        p_second = cit_second(0, np.arange(1, predictors.shape[1] + 1))
        if p_second < self.alpha_orientation < p_first:
            return '<-'
        if p_second > self.alpha_orientation > p_first:
            return '->'
        return '-'

    def _cross_derivatives(self, first_node: Any, second_node: Any, current_nodes: List[Any]) -> Tuple[NDArray, List[Any]]:
        current_nodes = list(np.sort(current_nodes))  # Sort for more cache hits
        X, score, kernel, s = self._get_score_and_helpers(current_nodes)
        first_idx = current_nodes.index(first_node)
        second_idx = current_nodes.index(second_node)
        first_col = self._hessian_col(X, score, first_idx, self.eta_h, kernel, s)[:, second_idx]
        second_col = self._hessian_col(X, score, second_idx, self.eta_h, kernel, s)[:, first_idx]
        return np.concatenate((first_col, second_col)), current_nodes

    def are_connected(self, first_node: Any, second_node: Any, current_nodes: List[Any]) -> bool:
        return ttest_1samp(self._cross_derivatives(first_node, second_node, current_nodes)[0], 0).pvalue < self.alpha_separations


    def prune_edge(self, first_node, second_node, neighbourhood) -> bool:
        deltas = Parallel(n_jobs=self.n_jobs)(
                delayed(lambda s: self._cross_derivatives(first_node, second_node, s + [first_node, second_node])
                        )(subset) for subset in _all_subsets(neighbourhood - {first_node, second_node})
        )


        delta_min, subset = min(deltas, key=lambda d: np.mean(np.abs(d[0])))
        return ttest_1samp(delta_min, 0).pvalue > self.alpha_separations


    def _cache_key(self, subset: List[Any]) -> str:
        return '.'.join(map(str, subset))

    def _get_score_and_helpers(self, subset: List[Any]) -> Tuple[NDArray, NDArray, NDArray, float]:
        key = self._cache_key(subset)
        if self.cache[key] is None:
            X = self.data[subset].to_numpy()
            _, d = X.shape
            s = np.max([kernel_width(X), self.score_eps])
            kernel = self._evaluate_kernel(X, s=s)
            nablaK = self._evaluate_nablaK(kernel, X, s)
            score = self.score(X, self.eta_g, kernel, nablaK)
            if self.use_cache:
                self.cache[key] = X, score, kernel, s
            else:
                return X, score, kernel, s
        return self.cache[key]

    def _cv_predict(self, X: NDArray, y: NDArray) -> NDArray:
        if self.cv == 1:
            return self.get_regression(X.shape[1]).fit(X, y).predict(X)
        predictions = []
        for train_index, test_index in KFold(n_splits=self.cv, shuffle=False).split(X, y):
            reg = self.get_regression(X.shape[1]).fit(X[train_index, :], y[train_index])
            predictions.append(y[test_index] - reg.predict(X[test_index, :]))
        return np.concatenate(predictions, axis=0)

