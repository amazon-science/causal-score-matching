# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

import os
import sys
import unittest

import networkx as nx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from benchmark.utils.metrics import dtop


class MetricsTest(unittest.TestCase):

    def test_dtop_correct_on_complete_dag(self):
        dag = nx.DiGraph()
        nodes = ['a', 'b', 'c', 'd']
        dag.add_nodes_from(nodes)
        for i, first in enumerate(nodes):
            for j, second in enumerate(nodes):
                if i < j:
                    dag.add_edge(first, second)

        self.assertEqual(0, dtop(dag, nodes))
        self.assertEqual(1, dtop(dag, ['a', 'b', 'd', 'c']))
        self.assertEqual(6, dtop(dag, ['d', 'c', 'b', 'a']))

    def test_dtop_correct_on_confounding_graph(self):
        dag = nx.DiGraph()
        dag.add_edge('a', 'b')
        dag.add_edge('a', 'c')

        self.assertEqual(0, dtop(dag, ['a', 'b', 'c']))
        self.assertEqual(0, dtop(dag, ['a', 'c', 'b']))
        self.assertEqual(1, dtop(dag, ['b', 'a', 'c']))
        self.assertEqual(2, dtop(dag, ['c', 'b', 'a']))

    def test_dtop_ignore_bidirected_correct_on_instrument_graph(self):
        dag = nx.DiGraph()
        dag.add_edge('i', 'x')
        dag.add_edge('x', 'y')
        dag.add_edge('y', 'x')

        self.assertEqual(0, dtop(dag, ['i', 'x', 'y'], ignore_bidirected=True))
        self.assertEqual(0, dtop(dag, ['i', 'y', 'x'], ignore_bidirected=True))
        self.assertEqual(1, dtop(dag, ['y', 'x', 'i'], ignore_bidirected=True))

        # Does this behaviour make sense?
        self.assertEqual(1, dtop(dag, ['i', 'x', 'y'], ignore_bidirected=False))
        self.assertEqual(1, dtop(dag, ['i', 'y', 'x'], ignore_bidirected=False))


if __name__ == '__main__':
    unittest.main()
