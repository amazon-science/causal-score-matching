# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

from collections import defaultdict
from typing import Tuple, Optional, List

import networkx as nx
import numpy as np
import pandas as pd
from numpy._typing import NDArray


class DataGenerator:

    def __init__(self,
                 num_observed_nodes: int = 8,
                 num_hidden: Optional[int] = None,
                 erdos_p: float = None,
                 mechanism: str = 'cam',
                 graph_type: str = 'dag',
                 coeff_range: Tuple[float, float] = (-1, 1)):
        if graph_type != 'dag':
            raise NotImplementedError('Graph Type can only be "dag" right now. Not ' + str(graph_type))
        self.num_observed_nodes = num_observed_nodes
        num_nodes = num_observed_nodes + num_hidden
        self.num_nodes = num_nodes
        self.nodes = ['V{}'.format(i + 1) for i in range(num_nodes)]
        self.p = (1.1 * np.log(num_nodes)) / num_nodes if erdos_p is None else erdos_p
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.nodes)
        self.causal_order = np.random.permutation(self.nodes)
        self.parent_idxs = defaultdict(lambda: [])
        self.mechanism_type = mechanism
        df_idx = {n: i for (i, n) in enumerate(self.nodes)}
        for i, x in enumerate(self.causal_order):
            for j, y in enumerate(self.causal_order):
                if j > i and np.random.rand() < self.p:
                    self.parent_idxs[y].append(df_idx[x])
                    self.graph.add_edge(x, y)
        nodes_with_neighbours = [df_idx[n] for n in self.graph.nodes if len(set(self.graph.successors(n))) > 0 and
                                 len(set(self.graph.successors(n)).union(set(self.graph.predecessors(n)))) > 1]
        if len(nodes_with_neighbours) < num_hidden:
            print('WARNING: not enough nodes with more than one neighbour!')
            self.hidden_idx = set(np.random.choice(list(df_idx.values()), num_hidden, replace=False))
        else:
            self.hidden_idx = set(np.random.choice(nodes_with_neighbours, num_hidden, replace=False))
        self.observed_idx = set(df_idx.values()) - set(self.hidden_idx)

        self.mechanism = {}
        for node in self.parent_idxs.keys():
            if mechanism == 'linear':
                self.mechanism[node] = LinearMechanism(len(self.parent_idxs[node]), coeff_range)
            elif mechanism == 'nn':
                num_parents = len(self.parent_idxs[node])
                self.mechanism[node] = NNMechanism(num_parents, 20, coeff_range=coeff_range)
            elif mechanism == 'nn_non_add':
                num_parents = len(self.parent_idxs[node]) + 1
                self.mechanism[node] = NNMechanism(num_parents, 20, coeff_range=coeff_range)
            elif mechanism == 'cam':
                num_parents = len(self.parent_idxs[node])
                self.mechanism[node] = CAMMechanism(num_parents, 20, coeff_range=coeff_range)
            elif mechanism == 'camsq':
                num_parents = len(self.parent_idxs[node])
                self.mechanism[node] = CAMPolyMechanism(num_parents)
            elif mechanism == 'ram':
                local_obs_idx = [i for (i, global_idx) in enumerate(self.parent_idxs[node]) if
                                 global_idx in self.observed_idx]
                local_hid_idx = [i for (i, global_idx) in enumerate(self.parent_idxs[node]) if
                                 global_idx in self.hidden_idx]
                self.mechanism[node] = RAMMechanism(local_obs_idx, local_hid_idx, 20, coeff_range=coeff_range)
            else:
                raise NotImplementedError('Non-linear model not implemented yet')

    def generate(self, num_samples: int = 100, var: float = 1, noise: str = 'gaussian') -> Tuple[
        pd.DataFrame, nx.DiGraph]:
        sample = pd.DataFrame(np.zeros((num_samples, self.num_nodes)), columns=self.nodes)
        if noise == 'gaussian':
            n_func = lambda: np.random.normal(loc=0, scale=var, size=num_samples)
        elif noise == 'uniform':
            a = np.sqrt(3 * var)  # get var as variance
            n_func = lambda: np.random.uniform(low=-a, high=a, size=num_samples)
        else:
            raise NotImplementedError('Invalid noise parameter: {}'.format(noise))
        for node in self.causal_order:
            values = n_func()  # Right now only non-transformed noise for roots

            if node in self.mechanism:
                if self.mechanism_type == 'nn_non_add':
                    inp = np.concatenate([sample.iloc[:, self.parent_idxs[node]].to_numpy(),
                                          np.expand_dims(n_func(), axis=-1)],
                                         axis=1
                                         )
                    values = self.mechanism[node](inp)
                else:
                    values += self.mechanism[node](sample.iloc[:, self.parent_idxs[node]].to_numpy())
            sample.loc[:, node] = values
        sample = sample.iloc[:, list(self.observed_idx)]
        return sample / sample.std(), self.graph

class NewDataGenerator(DataGenerator):

    def __init__(self, num_observed_nodes: int = 8, p_confounder: float = 0.2, p_mediator: float = .2, erdos_p: float = None,
                 mechanism: str = 'cam', graph_type: str = 'dag', coeff_range: Tuple[float, float] = (-1, 1)):
        super().__init__(num_observed_nodes, 0, erdos_p, mechanism, graph_type, coeff_range)
        new_edges = []
        for x, y in self.graph.edges:
            if np.random.rand() < .5:
                if np.random.rand() < p_confounder:
                    new_idx = self.num_nodes
                    self.num_nodes += 1
                    confounder_name = f'L{self.num_nodes}'
                    self.nodes.append(confounder_name)
                    self.hidden_idx.add(new_idx)
                    self.parent_idxs[x].append(new_idx)
                    self.parent_idxs[y].append(new_idx)
                    new_edges.append((confounder_name, x))
                    new_edges.append((confounder_name, y))
            else:
                if np.random.rand() < p_mediator:
                    new_idx = self.num_nodes
                    self.num_nodes += 1
                    mediator_name = f'M{self.num_nodes}'
                    self.nodes.append(mediator_name)
                    self.hidden_idx.add(new_idx)
                    self.parent_idxs[y].append(new_idx)
                    self.parent_idxs[mediator_name].append(self.nodes.index(x))
                    new_edges.append((mediator_name, y))
                    new_edges.append((x, mediator_name))
        self.graph.add_edges_from(new_edges)

        self.mechanism = {}
        for node in self.parent_idxs.keys():
            if mechanism == 'linear':
                self.mechanism[node] = LinearMechanism(len(self.parent_idxs[node]), coeff_range)
            elif mechanism == 'nn':
                num_parents = len(self.parent_idxs[node])
                self.mechanism[node] = NNMechanism(num_parents, 20, coeff_range=coeff_range)
            elif mechanism == 'nn_non_add':
                num_parents = len(self.parent_idxs[node]) + 1
                self.mechanism[node] = NNMechanism(num_parents, 20, coeff_range=coeff_range)
            elif mechanism == 'cam':
                num_parents = len(self.parent_idxs[node])
                self.mechanism[node] = CAMMechanism(num_parents, 20, coeff_range=coeff_range)
            elif mechanism == 'camsq':
                num_parents = len(self.parent_idxs[node])
                self.mechanism[node] = CAMPolyMechanism(num_parents)
            elif mechanism == 'ram':
                local_obs_idx = [i for (i, global_idx) in enumerate(self.parent_idxs[node]) if
                                 global_idx in self.observed_idx]
                local_hid_idx = [i for (i, global_idx) in enumerate(self.parent_idxs[node]) if
                                 global_idx in self.hidden_idx]
                self.mechanism[node] = RAMMechanism(local_obs_idx, local_hid_idx, 20, coeff_range=coeff_range)
            else:
                raise NotImplementedError('Non-linear model not implemented yet')


class LinearMechanism:
    def __init__(self, num_parents: int, coeff_range: Tuple[float, float], weights: NDArray = None):
        if weights is None:
            self.weights = np.random.uniform(low=coeff_range[0], high=coeff_range[1], size=num_parents) \
                           * np.random.choice([-1, 1])
        else:
            self.weights = weights

    def __call__(self, parents: NDArray) -> NDArray:
        return np.dot(self.weights, parents.T).T


class NNMechanism:
    def __init__(self, num_parents: int, num_hidden: int = 10, coeff_range: Tuple[float, float] = (-1, 1)):
        self.weights_in = np.random.uniform(low=coeff_range[0], high=coeff_range[1], size=(num_hidden, num_parents))
        self.bias = np.random.uniform(low=coeff_range[0], high=coeff_range[1])
        self.weights_out = np.random.uniform(low=-coeff_range[0], high=coeff_range[1], size=num_hidden)

    def __call__(self, parents: NDArray) -> NDArray:
        hidden = np.dot(self.weights_in, parents.T) + self.bias
        transformed = np.tanh(hidden)
        return np.dot(self.weights_out, transformed).T


class CAMMechanism:
    def __init__(self, num_parents: int, num_hidden: int = 10, coeff_range: Tuple[float, float] = (-1, 1)):
        self.mechanisms = []
        for _ in range(num_parents):
            self.mechanisms.append(NNMechanism(1, num_hidden, coeff_range))

    def __call__(self, parents: NDArray) -> NDArray:
        output = np.zeros(parents.shape[0])
        for i in range(parents.shape[1]):
            output += self.mechanisms[i](np.expand_dims(parents[:, i], -1))
        return output


class RAMMechanism:
    def __init__(self,
                 observed_parents_idx: List[int],
                 hidden_parents_idx: List[int],
                 num_hidden: int = 10,
                 coeff_range: Tuple[float, float] = (-3, 3)):
        self.observed_parents_idx = observed_parents_idx
        self.hidden_parents_idx = hidden_parents_idx
        self.mechanisms_observed = NNMechanism(len(observed_parents_idx), num_hidden, coeff_range)
        self.mechanisms_hidden = NNMechanism(len(hidden_parents_idx), num_hidden, coeff_range)

    def __call__(self, parents: NDArray) -> NDArray:
        output = np.zeros(parents.shape[0])
        output += self.mechanisms_observed(parents[:, self.observed_parents_idx])
        output += self.mechanisms_hidden(parents[:, self.hidden_parents_idx])
        return output


class CAMPolyMechanism:
    def __init__(self, num_parents: int, max_degree: int = 5, coeff_range: Tuple[float, float] = (-1, 1)):
        self.mechanisms = []
        for _ in range(num_parents):
            coefs = np.random.uniform(low=coeff_range[0], high=coeff_range[1], size=max_degree)
            self.mechanisms.append(lambda pa: np.sum([coefs[i] * pa ** i for i in range(max_degree)], axis=0)[:, 0])

    def __call__(self, parents: NDArray) -> NDArray:
        output = np.zeros(parents.shape[0])
        for i in range(parents.shape[1]):
            output += self.mechanisms[i](np.expand_dims(parents[:, i], -1))
        return output
