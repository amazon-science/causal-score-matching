# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

import os
import sys
import unittest

import numpy as np
import pandas as pd
from flaky import flaky

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from benchmark.data.generator import CAMPolyMechanism
from causal_discovery.adascore import AdaScore

ALPHA = 0.1
N_SAMPLES = 2000
NUM_RUNS = 20
REGRESSION = 'kernel_ridge'
CV = 3


class IsLeafTest(unittest.TestCase):

    @flaky(max_runs=3)
    def test_orient_chain(self):
        incorrect = []
        for _ in range(NUM_RUNS):
            x = np.random.uniform(size=(N_SAMPLES, 1))
            n_y = np.random.uniform(size=N_SAMPLES)
            n_z = np.random.uniform(size=N_SAMPLES)
            y = np.expand_dims(CAMPolyMechanism(1)(x) + n_y, axis=1)
            z = np.expand_dims(CAMPolyMechanism(1)(y) + n_z, axis=1)
            data = np.concatenate([x, y, z], axis=1)
            nodes = ['x', 'y', 'z']
            df = pd.DataFrame(data, columns=nodes)
            df = df / df.std()

            algo = AdaScore(ALPHA, ALPHA, ALPHA, regression=REGRESSION, cv=CV)
            algo._residual_oracle[frozenset(['x', 'y', 'z']), 'z'] = n_z
            algo._residual_oracle[frozenset(['y', 'z']), 'z'] = n_z
            algo._residual_oracle[frozenset(['x', 'y']), 'y'] = n_y
            algo.data = df
            res = algo.orient_edge('x', 'y', {'y'}, {'x', 'z'})
            incorrect.append(res != '->')
        self.assertLess(np.mean(incorrect), .5)

    def test_non_oriented_hidden_confounders(self):
        incorrect = []
        for _ in range(NUM_RUNS):
            x = np.random.uniform(size=(N_SAMPLES, 1))
            y = np.expand_dims(CAMPolyMechanism(1)(x) + np.random.uniform(size=N_SAMPLES), axis=1)
            z = np.expand_dims(CAMPolyMechanism(1)(x) + np.random.uniform(size=N_SAMPLES), axis=1)
            data = np.concatenate([y, z], axis=1)
            nodes = ['y', 'z']
            df = pd.DataFrame(data, columns=nodes)
            df = df / df.std()

            algo = AdaScore(ALPHA, ALPHA, ALPHA, regression=REGRESSION, cv=CV)
            algo.data = df
            res = algo.orient_edge('y', 'z', {'z'}, {'y'})
            incorrect.append(res != '-')
        self.assertLess(np.mean(incorrect), .5)

    def test_non_leaf_hidden_mediator(self):
        incorrect = []
        for _ in range(NUM_RUNS):
            x = np.random.uniform(size=(N_SAMPLES, 1))
            y = np.expand_dims(CAMPolyMechanism(1)(x) + np.random.uniform(size=N_SAMPLES), axis=1)
            z = np.expand_dims(CAMPolyMechanism(1)(y) + np.random.uniform(size=N_SAMPLES), axis=1)
            data = np.concatenate([x, z], axis=1)
            nodes = ['x', 'z']
            df = pd.DataFrame(data, columns=nodes)
            df = df / df.std()

            algo = AdaScore(ALPHA, ALPHA, ALPHA, regression=REGRESSION, cv=CV)
            algo.data = df
            res = algo.orient_edge('x', 'z', {'z'}, {'x'})
            incorrect.append(res != '-')
        self.assertLess(np.mean(incorrect), .5)
    # TODO test doesn't work


if __name__ == '__main__':
    unittest.main()
