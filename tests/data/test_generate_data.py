# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

import os
import sys
import unittest

import networkx as nx
import numpy as np
from cdt.metrics import SHD

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from benchmark.data.generate_data import marginalize, get_confounded_datasets, get_pag_skel_with_ada_orientations


class GenerateDataTest(unittest.TestCase):

    def test_data_generation_correct_shape(self):
        NUM_NODES = 10
        NUM_HIDDEN = 3
        NUM_SAMPLES = 100
        dataset, graph = get_confounded_datasets(1, NUM_NODES, NUM_HIDDEN, NUM_SAMPLES)[0]
        self.assertEqual(NUM_NODES, len(dataset.keys()))
        self.assertEqual(NUM_SAMPLES, dataset.shape[0])
        self.assertEqual(NUM_NODES + NUM_HIDDEN, len(graph.nodes))

    def test_marginalize_returns_expected_graph_for_m_graph(self):
        dag = nx.DiGraph()
        dag.add_edge('a', 'x')
        dag.add_edge('a', 'b')
        dag.add_edge('c', 'b')
        dag.add_edge('c', 'y')

        subset = {'x', 'y'}
        marginal_dag = marginalize(dag, subset)
        self.assertTrue(np.all(subset == set(marginal_dag.nodes)))

        expected_dag = nx.DiGraph()
        expected_dag.add_nodes_from(['x', 'y'])
        self.assertEqual(0, SHD(expected_dag, marginal_dag))
        marginal_dag = marginalize(dag, subset, indicate_confounding=False)
        self.assertEqual(0, SHD(expected_dag, marginal_dag))

        expected_dag.add_edge('x', 'y')
        self.assertNotEqual(0, SHD(expected_dag, marginal_dag))

    def test_marginalize_returns_expected_graph_for_chain_graph(self):
        dag = nx.DiGraph()
        dag.add_edge('a', 'b')
        dag.add_edge('b', 'c')
        dag.add_edge('c', 'd')

        subset = {'a', 'c'}
        marginal_dag = marginalize(dag, subset)
        self.assertTrue(np.all(subset == set(marginal_dag.nodes)))

        expected_dag = nx.DiGraph()
        expected_dag.add_edge('a', 'c')
        self.assertEqual(0, SHD(expected_dag, marginal_dag))

        marginal_dag = marginalize(dag, subset, indicate_confounding=False)
        self.assertEqual(0, SHD(expected_dag, marginal_dag))

        marginal_dag = marginalize(dag, subset, indicate_unobs_direct_paths=True)
        expected_dag.add_edge('c', 'a')  # This is a UDP as defined in Maeda et al. (2022)
        self.assertEqual(0, SHD(expected_dag, marginal_dag))

        marginal_dag = marginalize(dag, subset)
        expected_dag.remove_edge('a', 'c')
        self.assertNotEqual(0, SHD(expected_dag, marginal_dag))

    def test_marginalize_returns_expected_graph_for_unshielded_collider_graph(self):
        dag = nx.DiGraph()
        dag.add_edge('a', 'b')
        dag.add_edge('c', 'b')
        dag.add_edge('d', 'c')

        subset = {'a', 'c'}
        marginal_dag = marginalize(dag, subset)
        self.assertTrue(np.all(subset == set(marginal_dag.nodes)))

        expected_dag = nx.DiGraph()
        expected_dag.add_nodes_from(['a', 'c'])
        self.assertEqual(0, SHD(expected_dag, marginal_dag))

        marginal_dag = marginalize(dag, subset, indicate_unobs_direct_paths=True)
        self.assertEqual(0, SHD(expected_dag, marginal_dag))

        marginal_dag = marginalize(dag, subset, indicate_confounding=False)
        self.assertEqual(0, SHD(expected_dag, marginal_dag))

        expected_dag.add_edge('a', 'c')
        self.assertNotEqual(0, SHD(expected_dag, marginal_dag))

    def test_marginalize_returns_expected_graph_for_pure_confounding_graph(self):
        dag = nx.DiGraph()
        dag.add_edge('a', 'b')
        dag.add_edge('a', 'c')

        subset = {'b', 'c'}
        marginal_dag = marginalize(dag, subset)
        self.assertTrue(np.all(subset == set(marginal_dag.nodes)))

        expected_dag = nx.DiGraph()
        expected_dag.add_nodes_from(['b', 'c'])
        expected_dag.add_edge('b', 'c')
        expected_dag.add_edge('c', 'b')
        self.assertEqual(0, SHD(expected_dag, marginal_dag))

        marginal_dag = marginalize(dag, subset, indicate_unobs_direct_paths=True)
        self.assertEqual(0, SHD(expected_dag, marginal_dag))

        marginal_dag = marginalize(dag, subset, indicate_confounding=False)
        expected_dag.remove_edge('b', 'c')
        expected_dag.remove_edge('c', 'b')
        self.assertEqual(0, SHD(expected_dag, marginal_dag))

        marginal_dag = marginalize(dag, subset)
        expected_dag.add_edge('b', 'c')
        self.assertNotEqual(0, SHD(expected_dag, marginal_dag))

    def test_get_confounded_datasets_outputs_correct_expected_degree(self):
        expected_degree = 2
        datasets = get_confounded_datasets(num_datasets=1000,
                                           num_nodes=10,
                                           num_hidden=0,
                                           num_samples=3,
                                           expected_out_degree=expected_degree
                                           )
        out_degrees = []
        for _, graph in datasets:
            for node in graph.nodes:
                out_degrees.append(len(list(graph.successors(node))))
        self.assertAlmostEquals(np.mean(out_degrees), expected_degree, places=1)

    def test_get_confounded_datasets_outputs_correct_num_dimensions(self):
        num_nodes = 10
        num_hidden = 5
        num_samples = 100
        datasets = get_confounded_datasets(10, num_nodes, num_hidden, num_samples)
        for data, graph in datasets:
            self.assertEqual(len(list(graph.nodes)), num_nodes + num_hidden)
            self.assertEqual(len(data.keys()), num_nodes)
            self.assertEqual(data.shape[0], num_samples)

    def test_pag_skel_and_ada_dir_unshielded_collider(self):
        dag = nx.DiGraph()
        dag.add_edge('a', 'b')
        dag.add_edge('c', 'b')
        dag.add_edge('d', 'c')
        subset = ['a', 'c']
        res_dag = get_pag_skel_with_ada_orientations(dag, subset)
        self.assertTrue(np.all(set(subset) == set(res_dag.nodes)))

        expected_dag = nx.DiGraph()
        expected_dag.add_nodes_from(subset)
        self.assertEqual(0, SHD(expected_dag, res_dag))

    def test_pag_skel_and_ada_dir_pure_confounder(self):
        dag = nx.DiGraph()
        dag.add_edge('a', 'b')
        dag.add_edge('a', 'c')

        subset = ['b', 'c']
        res_dag = get_pag_skel_with_ada_orientations(dag, subset)
        self.assertTrue(np.all(set(subset) == set(res_dag.nodes)))

        expected_dag = nx.DiGraph()
        expected_dag.add_edge('b', 'c')
        expected_dag.add_edge('c', 'b')
        self.assertEqual(0, SHD(expected_dag, res_dag))

    def test_pag_skel_and_ada_dir_instrument(self):
        dag = nx.DiGraph()
        dag.add_edge('i', 'x')
        dag.add_edge('z', 'x')
        dag.add_edge('z', 'y')
        dag.add_edge('x', 'y')

        subset = ['i', 'x', 'y']
        res_dag = get_pag_skel_with_ada_orientations(dag, subset)
        self.assertTrue(np.all(set(subset) == set(res_dag.nodes)))

        expected_dag = nx.DiGraph()
        expected_dag.add_edge('i', 'x')
        expected_dag.add_edge('x', 'y')
        expected_dag.add_edge('y', 'x')
        expected_dag.add_edge('i', 'y')
        expected_dag.add_edge('y', 'i')
        self.assertEqual(0, SHD(expected_dag, res_dag))

    def test_pag_skel_and_ada_dir_instrument_plus_direct(self):
        dag = nx.DiGraph()
        dag.add_edge('i', 'x')
        dag.add_edge('z', 'x')
        dag.add_edge('z', 'y')
        dag.add_edge('x', 'y')
        dag.add_edge('i', 'y')

        subset = ['i', 'x', 'y']
        res_dag = get_pag_skel_with_ada_orientations(dag, subset)
        self.assertTrue(np.all(set(subset) == set(res_dag.nodes)))

        expected_dag = nx.DiGraph()
        expected_dag.add_edge('i', 'x')
        expected_dag.add_edge('x', 'y')
        expected_dag.add_edge('y', 'x')
        expected_dag.add_edge('i', 'y')
        expected_dag.add_edge('y', 'i')
        self.assertEqual(0, SHD(expected_dag, res_dag))


if __name__ == '__main__':
    unittest.main()
