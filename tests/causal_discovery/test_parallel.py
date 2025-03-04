# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

import os
import random
import sys
import unittest

import numpy as np
import torch

from benchmark.utils.algo_wrappers import AdaScoreWrapper

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from benchmark.data.generate_data import get_confounded_datasets


class ParallelAdaTest(unittest.TestCase):

    def test_parallel_random_graphs(self):
        seed = 42
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        for i, (data, ground_truth) in enumerate(get_confounded_datasets(10, 7, 1, 500)):
            print("Test dataset Nr. {}".format(i))
            original_algo = AdaScoreWrapper()
            parallel_algo = AdaScoreWrapper(n_jobs=2)
            g_original = original_algo.fit(data)
            g_parallel = parallel_algo.fit(data)
            shd = g_original.shd(g_parallel)
            self.assertEqual(0, shd)


if __name__ == '__main__':
    unittest.main()
