# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

import os
import sys
import unittest

import numpy as np
import pandas as pd
from dodiscover import make_context
from dodiscover.toporder import NoGAM
from flaky import flaky

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from benchmark.data.generator import CAMPolyMechanism
from causal_discovery.adascore import AdaScore

ALPHA = 0.1
N_SAMPLES = 1000
NUM_RUNS = 10
REGRESSION = 'kernel_ridge'
CV = 3


class IsLeafTest(unittest.TestCase):

    def test_leaf_collider_same_as_nogam(self):
        incorrect = []
        for _ in range(NUM_RUNS):
            x = np.random.uniform(size=(N_SAMPLES, 1))
            y = np.random.uniform(size=(N_SAMPLES, 1))
            nz = np.random.uniform(size=N_SAMPLES)
            z = np.expand_dims(CAMPolyMechanism(1)(x) + CAMPolyMechanism(1)(y) + nz, axis=1)
            data = np.concatenate([x, y, z], axis=1)
            nodes = ['x', 'y', 'z']
            df = pd.DataFrame(data, columns=nodes)
            df = df / df.std()

            nogam = NoGAM(n_crossval=CV)
            context = make_context().variables(data=df).build()
            nogam.learn_graph(df, context)
            nogam_leaf = nodes[nogam.order_[-1]]
            algo = AdaScore(ALPHA, ALPHA, ALPHA, regression=REGRESSION, cv=CV)
            for node in df.keys():
                algo.boundaries[node] = set(df.keys()) - {node}
            algo.data = df
            res, significant = algo.get_unconfounded_leaf(nodes, nodes)
            incorrect.append(res != nogam_leaf)
        self.assertEqual(np.sum(incorrect), 0)

    @flaky(max_runs=3)
    def test_leaf_collider(self):
        incorrect = []
        for _ in range(NUM_RUNS):
            x = np.random.uniform(size=(N_SAMPLES, 1))
            y = np.random.uniform(size=(N_SAMPLES, 1))
            nz = np.random.uniform(size=N_SAMPLES)
            z = np.expand_dims(CAMPolyMechanism(1)(x) + CAMPolyMechanism(1)(y) + nz, axis=1)
            data = np.concatenate([x, y, z], axis=1)
            nodes = ['x', 'y', 'z']
            df = pd.DataFrame(data, columns=nodes)
            df = df / df.std()

            algo = AdaScore(ALPHA, ALPHA, ALPHA, regression=REGRESSION, cv=CV)
            for node in df.keys():
                algo.boundaries[node] = set(df.keys()) - {node}
            algo._residual_oracle[frozenset(['x', 'y', 'z']), 'z'] = nz
            algo._residual_oracle[frozenset(['y', 'z']), 'z'] = x + nz
            algo._residual_oracle[frozenset(['x', 'z']), 'z'] = y + nz
            algo.data = df
            res, significant = algo.get_unconfounded_leaf(nodes, nodes)
            incorrect.append(res != 'z' or not significant)
        self.assertLess(np.mean(incorrect), .5)  # TODO why is this case so difficult?

    def test_leaf_chain_same_as_nogam(self):
        incorrect = []
        for _ in range(NUM_RUNS):
            x = np.random.uniform(size=(N_SAMPLES, 1))
            y = np.expand_dims(CAMPolyMechanism(1)(x) + np.random.uniform(size=N_SAMPLES), axis=1)
            z = np.expand_dims(CAMPolyMechanism(1)(y) + np.random.uniform(size=N_SAMPLES), axis=1)
            data = np.concatenate([x, y, z], axis=1)
            nodes = ['x', 'y', 'z']
            df = pd.DataFrame(data, columns=nodes)
            df = df / df.std()

            nogam = NoGAM(n_crossval=CV)
            context = make_context().variables(data=df).build()
            nogam.learn_graph(df, context)
            nogam_leaf = nodes[nogam.order_[-1]]
            algo = AdaScore(ALPHA, ALPHA, ALPHA, regression=REGRESSION, cv=CV)
            for node in df.keys():
                algo.boundaries[node] = set(df.keys()) - {node}
            algo.data = df
            res, significant = algo.get_unconfounded_leaf(nodes, nodes)
            incorrect.append(res != nogam_leaf)
        self.assertEquals(np.sum(incorrect), 0)

    @flaky(max_runs=3)
    def test_leaf_chain(self):
        incorrect = []
        for _ in range(NUM_RUNS):
            x = np.random.uniform(size=(N_SAMPLES, 1))
            n_y = np.random.uniform(size=N_SAMPLES)
            y = np.expand_dims(CAMPolyMechanism(1)(x) + n_y, axis=1)
            n_z = np.random.uniform(size=N_SAMPLES)
            z = np.expand_dims(CAMPolyMechanism(1)(y) + n_z, axis=1)
            data = np.concatenate([x, y, z], axis=1)
            nodes = ['x', 'y', 'z']
            df = pd.DataFrame(data, columns=nodes)
            df = df / df.std()

            algo = AdaScore(ALPHA, ALPHA, ALPHA, regression=REGRESSION, cv=CV)
            for node in df.keys():
                algo.boundaries[node] = set(df.keys()) - {node}
            algo._residual_oracle[frozenset(['x', 'y']), 'y'] = n_y
            algo._residual_oracle[frozenset(['x', 'y', 'z']), 'z'] = n_z
            algo._residual_oracle[frozenset(['y', 'z']), 'z'] = n_z
            algo.data = df
            res, significant = algo.get_unconfounded_leaf(nodes, nodes)
            incorrect.append(res != 'z' or not significant)
        self.assertLess(np.mean(incorrect), .5)

    def test_leaf_confounders_same_as_nogam(self):
        incorrect = []
        for _ in range(NUM_RUNS):
            x = np.random.uniform(size=(N_SAMPLES, 1))
            y = np.expand_dims(CAMPolyMechanism(1)(x) + np.random.uniform(size=N_SAMPLES), axis=1)
            xy_array = np.concatenate([x, y], axis=1)
            z = np.expand_dims(CAMPolyMechanism(2)(xy_array) + np.random.uniform(size=N_SAMPLES), axis=1)
            data = np.concatenate([x, y, z], axis=1)
            nodes = ['x', 'y', 'z']
            df = pd.DataFrame(data, columns=nodes)
            df = df / df.std()

            nogam = NoGAM(n_crossval=CV)
            context = make_context().variables(data=df).build()
            nogam.learn_graph(df, context)
            nogam_leaf = nodes[nogam.order_[-1]]
            algo = AdaScore(ALPHA, ALPHA, ALPHA, regression=REGRESSION, cv=CV)
            for node in df.keys():
                algo.boundaries[node] = set(df.keys()) - {node}
            algo.data = df
            res, significant = algo.get_unconfounded_leaf(nodes, nodes)
            incorrect.append(res != nogam_leaf)
        self.assertEqual(np.sum(incorrect), 0)

    @flaky(max_runs=3)
    def test_leaf_confounders(self):
        incorrect = []
        for _ in range(NUM_RUNS):
            x = np.random.uniform(size=(N_SAMPLES, 1))
            n_y = np.random.uniform(size=N_SAMPLES)
            y = np.expand_dims(CAMPolyMechanism(1)(x) + n_y, axis=1)
            xy_array = np.concatenate([x, y], axis=1)
            n_z = np.random.uniform(size=N_SAMPLES)
            z = np.expand_dims(CAMPolyMechanism(2)(xy_array) + n_z, axis=1)
            data = np.concatenate([x, y, z], axis=1)
            nodes = ['x', 'y', 'z']
            df = pd.DataFrame(data, columns=nodes)
            df = df / df.std()

            algo = AdaScore(ALPHA, ALPHA, ALPHA, regression=REGRESSION, cv=CV)
            for node in df.keys():
                algo.boundaries[node] = set(df.keys()) - {node}
            algo._residual_oracle[frozenset(['x', 'y', 'z']), 'z'] = n_z
            algo._residual_oracle[frozenset(['x', 'y']), 'y'] = n_y
            algo.data = df
            res, significant = algo.get_unconfounded_leaf(nodes, nodes)
            incorrect.append(res != 'z' or not significant)
        self.assertLess(np.mean(incorrect), .5)

    def test_non_leaf_hidden_confounders(self):
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
            for node in df.keys():
                algo.boundaries[node] = set(df.keys()) - {node}
            algo.data = df
            _, significant = algo.get_unconfounded_leaf(nodes, nodes)
            incorrect.append(significant)  # If any node is significant, this should be an error
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
            for node in df.keys():
                algo.boundaries[node] = set(df.keys()) - {node}
            algo.data = df
            _, significant = algo.get_unconfounded_leaf(nodes, nodes)
            incorrect.append(significant)  # If any node is significant, this should be an error
        self.assertLess(np.mean(incorrect), .5)


if __name__ == '__main__':
    unittest.main()
