# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

import os
import sys
import unittest

import networkx as nx
from causallearn.utils.cit import CIT
from flaky import flaky

from benchmark.data.generator import DataGenerator
from causal_discovery.adascore import AdaScore
from causal_discovery.score_independence import ScoreIndependence

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

ALPHA = 0.05
N_SAMPLES = 1000
NUM_RUNS = 20


class SeparationTest(unittest.TestCase):
    @flaky(max_runs=3)
    def test_score_independences(self):
        false_dep = 0
        num_tests = 0
        for _ in range(NUM_RUNS):
            g = DataGenerator(num_observed_nodes=4, num_hidden=0)
            sample, gt = g.generate(num_samples=500)
            nodes = list(g.nodes)

            algo = AdaScore(ALPHA, ALPHA, ALPHA)
            algo.data = sample
            algo.cit = CIT(sample.to_numpy(), 'kci')
            algo.cit_idx = {n: i for i, n in enumerate(nodes)}
            for x in nodes:
                for y in nodes:
                    if x != y:
                        d_sep = nx.d_separated(gt, {x}, {y}, set([]))
                        ind = not algo.are_connected(x, y, [x, y])
                        num_tests += 1
                        if not ind and d_sep or ind and not d_sep:
                            false_dep += 1
        print(false_dep / float(num_tests))
        self.assertLess(false_dep / float(num_tests), .5)

    def test_cit_class(self):
        false_dep = 0
        fp = 0
        fn = 0
        num_tests = 0
        for _ in range(NUM_RUNS):
            g = DataGenerator(num_nodes=4)
            sample, gt = g.generate(num_samples=500)
            nodes = list(g.nodes)

            cit = ScoreIndependence(sample.to_numpy())
            for i, x in enumerate(nodes):
                for j, y in enumerate(nodes):
                    if x != y:
                        d_sep = nx.d_separated(gt, {x}, {y}, set([]))
                        ind = cit(i, j, []) > ALPHA
                        num_tests += 1
                        if not ind and d_sep or ind and not d_sep:
                            if ind and not d_sep:
                                fn += 1
                            elif not ind and d_sep:
                                fp += 1
                            false_dep += 1
        print(false_dep / float(num_tests), fp / float(num_tests), fn / float(num_tests))
        self.assertLess(false_dep / float(num_tests), .5)


if __name__ == '__main__':
    unittest.main()
