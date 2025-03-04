# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

import os
import random
import sys
import unittest

import networkx as nx
import numpy as np
import pandas as pd
import torch
from cdt.metrics import SHD
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from causal_discovery.oracle_adascore import OracleAdaScore
from benchmark.data.generate_data import get_confounded_datasets, get_pag_skel_with_ada_orientations


class OracleTest(unittest.TestCase):

    def test_oracle_m_graph(self):
        dag = nx.DiGraph()
        dag.add_edge('a', 'x')
        dag.add_edge('a', 'b')
        dag.add_edge('c', 'b')
        dag.add_edge('c', 'y')

        est_dag = OracleAdaScore(dag).fit(pd.DataFrame(columns=list(dag.nodes)))
        self.assertEqual(0, SHD(dag, est_dag))

    def test_marginalisation_chain(self):
        dag = nx.DiGraph()
        dag.add_edge('a', 'b')
        dag.add_edge('b', 'c')
        dag.add_edge('c', 'd')

        est_dag = OracleAdaScore(dag).fit(pd.DataFrame(columns=list(dag.nodes)))
        self.assertEqual(0, SHD(dag, est_dag))

    def test_marginalisation_collider(self):
        dag = nx.DiGraph()
        dag.add_edge('a', 'b')
        dag.add_edge('c', 'b')
        dag.add_edge('d', 'c')

        est_dag = OracleAdaScore(dag).fit(pd.DataFrame(columns=list(dag.nodes)))
        self.assertEqual(0, SHD(dag, est_dag))

    def test_oracle_confounders(self):
        dag = nx.DiGraph()
        dag.add_edge('c', 'x')
        dag.add_edge('c', 'y')

        observed = ['x', 'y']
        est_dag = OracleAdaScore(dag).fit(pd.DataFrame(columns=list(observed)))
        self.assertEqual(0, SHD(get_pag_skel_with_ada_orientations(dag, observed), est_dag))

    def test_oracle_instrument_variable(self):
        dag = nx.DiGraph()
        dag.add_edge('c', 'x')
        dag.add_edge('c', 'y')
        dag.add_edge('x', 'y')
        dag.add_edge('i', 'x')

        observed = ['i', 'x', 'y']
        est_dag = OracleAdaScore(dag).fit(pd.DataFrame(columns=list(observed)))
        self.assertEqual(0, SHD(get_pag_skel_with_ada_orientations(dag, observed), est_dag))

    def test_oracle_random_graphs(self):
        seed = 42
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        for i, (data, ground_truth) in enumerate(get_confounded_datasets(10, 10, 1, 3)):
            print("Test dataset Nr. {}".format(i))
            algo = OracleAdaScore(ground_truth)
            g_hat = algo.fit(data)
            marginal_ground_truth = get_pag_skel_with_ada_orientations(ground_truth, data.keys())
            s = SHD(marginal_ground_truth, g_hat)
            if s != 0:
                print(s)
                pos = nx.spring_layout(ground_truth, k=2)
                _, axs = plt.subplots(1, 3)
                nx.draw(ground_truth, pos=pos, ax=axs[0], with_labels=True)
                nx.draw(g_hat, pos=pos, ax=axs[1], with_labels=True)
                nx.draw(marginal_ground_truth, pos=pos, with_labels=True, ax=axs[2])
                plt.show()
            self.assertEqual(0, s)


if __name__ == '__main__':
    unittest.main()
