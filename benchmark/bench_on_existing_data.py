# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

import argparse
import glob
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmark.utils.causal_graphs import MixedGraph
from utils.algo_wrappers import get_algo_from_name

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Experiment with increasing node size.')
    parser.add_argument('--algorithms', nargs='+', default=['ridge'])
    parser.add_argument('--alpha_others', default=0.01, type=float)
    parser.add_argument('--alpha_confounded_leaf', default=0.05, type=float)
    parser.add_argument('--alpha_orientations', default=0.05, type=float)
    parser.add_argument('--alpha_separations', default=.05, type=float)
    parser.add_argument('--alpha_cam', default=0.001, type=float)  # The usual value in CAM
    parser.add_argument('--dir', default='.')
    parser.add_argument('--cv', default=3, type=int)
    parser.add_argument('--n_jobs', default=2, type=int)
    params = vars(parser.parse_args())

    with open(os.path.join(params['dir'], 'params.json')) as file:
        experiment_params = json.load(file)

    results = defaultdict(lambda: [])
    for data_subdir in glob.glob(os.path.join(params['dir'], 'data', '*')):
        # structure is e.g. './data/num_nodes_i/data_j.csv'. We want 'num_nodes_i' to insert in other paths later
        subdir = os.path.basename(os.path.normpath(data_subdir))
        logging.info("Enter subdir '{}'".format(subdir))
        def worker(i):
            local_result = {}
            logging.info("Test dataset Nr. {}".format(i))
            graph_dir = os.path.join(params['dir'], 'graphs', subdir)
            ground_truth = MixedGraph.load_graph(os.path.join(graph_dir, 'ground_truth_{}.gml'.format(i)))
            data = pd.read_csv(os.path.join(params['dir'], 'data', subdir, 'data_{}.csv'.format(i)), index_col=0)
            for algo_name in params['algorithms']:

                algo = get_algo_from_name(algo_name, params)

                start_t = time.time()
                g_hat = algo.fit(data)
                end_t = time.time()
                indicate_ucp = experiment_params['mechanism'] != 'linear' if 'mechanism' in experiment_params else True
                metrics = g_hat.eval_all_metrics(ground_truth.graph, indicate_ucp)
                metrics['time'] = end_t - start_t
                metrics['num_nodes'] = len(data.keys())
                # results[algo_name].append(metrics)
                g_hat.save_graph(os.path.join(graph_dir, 'g_hat_{}_{}.gml'.format(algo_name, i)))
                local_result[algo_name] = metrics
            return local_result


        result = Parallel(n_jobs=params['n_jobs'])(
            delayed(worker)(i) for i in range(experiment_params['num_datasets']))

        for algo_name in params['algorithms']:
            results[algo_name] += [r[algo_name] for r in result]

    for algo_name in params['algorithms']:
        df = pd.DataFrame(results[algo_name])
        df.to_csv(os.path.join(params['dir'], algo_name + '.csv'))
