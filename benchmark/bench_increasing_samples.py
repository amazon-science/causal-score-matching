# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmark.bench_increasing_size import get_causally_datasets
from benchmark.utils.causal_graphs import MixedGraph
from utils.algo_wrappers import get_algo_from_name
from utils.cache_source_files import copy_referenced_files_to

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Experiment with increasing sample size.')
    parser.add_argument('--algorithms',
                        nargs='+',
                        default=['ridge', 'camuv', 'nogam', 'rcd', 'lingam', 'random', 'fully_random']
                        )
    parser.add_argument('--num_samples', nargs='+', default=[500, 1000, 1500, 2000])
    parser.add_argument('--num_hidden', default=2, type=int)
    parser.add_argument('--num_nodes', default=7, type=int)
    parser.add_argument('--num_datasets', default=20, type=int)
    parser.add_argument('--p_edge', default=None, type=float)
    parser.add_argument('--m_edge', default=None, type=int)
    parser.add_argument('--noise', default="uniform", type=str)
    parser.add_argument('--alpha_others', default=0.05, type=float)
    parser.add_argument('--alpha_confounded_leaf', default=0.05, type=float)  # low alpha: enter in nogam regime
    parser.add_argument('--alpha_orientations', default=0.05, type=float)
    parser.add_argument('--alpha_separations', default=0.05, type=float)  # high: no pre-pruning.
    parser.add_argument('--alpha_cam', default=0.001, type=float)  # The usual value in CAM
    parser.add_argument('--scm', default='nonlinear')
    parser.add_argument('--cv', default=3, type=int)
    parser.add_argument('--n_jobs_datasets', default=2, type=int)
    parser.add_argument('--n_jobs_ada', default=1, type=int)
    params = vars(parser.parse_args())

    if params['n_jobs_datasets'] > 1 and params['n_jobs_ada'] > 1:
        print('WARNING: Having both, the experiment loop and adascore, parallel is untested.')

    result_dir = os.path.join('benchmark', 'logs', 'paper-plots', 'incr_samples_') + time.strftime('%y.%m.%d_%H.%M.%S')
    Path(result_dir).mkdir(parents=True, exist_ok=False)
    with open(os.path.join(result_dir, 'params.json'), 'w') as file:
        json.dump(params, file)
    copy_referenced_files_to(__file__, os.path.join(result_dir, "src_dump"))

    results = defaultdict(lambda: [])
    for num_samples in params['num_samples']:
        graph_dir = os.path.join(result_dir, 'graphs', 'num_samples_{}'.format(num_samples))
        data_dir = os.path.join(result_dir, 'data', 'num_samples_{}'.format(num_samples))
        Path(graph_dir).mkdir(parents=True, exist_ok=False)
        Path(data_dir).mkdir(parents=True, exist_ok=False)
        datasets = get_causally_datasets(
                    num_datasets=params['num_datasets'],
                    num_samples=int(num_samples),
                    num_observed_nodes=params['num_nodes'],
                    num_hidden=params["num_hidden"],
                    scm_type=params["scm"],
                    noise_dist=params["noise"],
                    p_edge=params["p_edge"],
                    expected_degree=params["m_edge"],
                    standardize=True
                )
        def worker(i, ground_truth, data):
            local_result = {}
            print(f"\n##############\nDataset {i}")
            logging.info("Test dataset Nr. {}".format(i))
            MixedGraph(ground_truth).save_graph(os.path.join(graph_dir, 'ground_truth_{}.gml'.format(i)))
            data.to_csv(os.path.join(data_dir, 'data_{}.csv'.format(i)))
            for algo_name in params['algorithms']:

                algo = get_algo_from_name(algo_name, params)

                start_t = time.time()
                g_hat = algo.fit(data)
                end_t = time.time()

                metrics = g_hat.eval_all_metrics(ground_truth, params['scm'] != 'linear')
                metrics['time'] = end_t - start_t
                metrics['num_samples'] = num_samples
                # results[algo_name].append(metrics)
                g_hat.save_graph(os.path.join(graph_dir, 'g_hat_{}_{}.gml'.format(algo_name, i)))

                print(f"Algorithm {algo_name}: time = {metrics['time']}s; SHD = {metrics['shd']}; skeleton f1 = {metrics['skeleton_f1']}")
                local_result[algo_name] = metrics
            return local_result


        result = Parallel(n_jobs=params['n_jobs_datasets'])(
            delayed(worker)(i, ground_truth, data) for i, (data, ground_truth) in enumerate(datasets))

        for algo_name in params['algorithms']:
            results[algo_name] += [r[algo_name] for r in result]

    for algo_name in params['algorithms']:
        df = pd.DataFrame(results[algo_name])
        df.to_csv(os.path.join(result_dir, algo_name + '.csv'))
