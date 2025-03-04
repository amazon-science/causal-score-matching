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

import cdt.data
import networkx as nx
import numpy as np
import pandas as pd
import torch
from bnlearn import import_example, import_DAG
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.io import loadmat


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmark.data.generate_data import get_pag_skel_with_ada_orientations
from benchmark.utils.cache_source_files import copy_referenced_files_to
from benchmark.utils.causal_graphs import MixedGraph
from utils.algo_wrappers import get_algo_from_name


def introduce_confounders(df: pd.DataFrame, gt: nx.DiGraph, num_hidden: int, num_samples: int) -> pd.DataFrame:
    while(True):
        node_idx = np.random.choice(range(df.shape[1]), df.shape[1] - num_hidden, replace=False)
        nodes = np.array(df.keys())[node_idx]
        marginal_gt = get_pag_skel_with_ada_orientations(gt, nodes, True)
        num_bi_edges = len([(x, y) for x in marginal_gt.nodes for y in marginal_gt.nodes
                            if marginal_gt.has_edge(x, y) and marginal_gt.has_edge(y, x)])
        if num_bi_edges == 0:
            print("No confoudners introduced. Resample.")
            continue
        if df.shape[0] > 1000:
            sample_idx = np.random.choice(range(df.shape[0]), num_samples, replace=False)
        else:
            sample_idx = np.arange(df.shape[0])
        return df.iloc[sample_idx, node_idx]

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Experiment with increasing node size.')
    parser.add_argument('--algorithms', nargs='+', default=['ridge', 'camuv', 'nogam', 'rcd', 'lingam'])
    parser.add_argument('--alpha_others', default=0.01, type=float)
    parser.add_argument('--alpha_confounded_leaf', default=0.05, type=float)
    parser.add_argument('--alpha_orientations', default=0.05, type=float)
    parser.add_argument('--alpha_separations', default=.05, type=float)
    parser.add_argument('--alpha_cam', default=0.001, type=float)  # The usual value in CAM
    parser.add_argument('--dataset', default='sachs')
    parser.add_argument('--num_datasets', default=20, type=int)
    parser.add_argument('--num_samples', default=1000, type=int)
    parser.add_argument('--num_hidden', default=2, type=int)
    parser.add_argument('--cv', default=3, type=int)
    parser.add_argument('--n_jobs_datasets', default=2, type=int)
    parser.add_argument('--n_jobs_ada', default=1, type=int)
    params = vars(parser.parse_args())

    result_dir = os.path.join('benchmark', 'logs', 'paper-plots', 'real_data') + time.strftime('%y.%m.%d_%H.%M.%S')
    Path(result_dir).mkdir(parents=True, exist_ok=False)
    with open(os.path.join(result_dir, 'params.json'), 'w') as file:
        json.dump(params, file)
    copy_referenced_files_to(__file__, os.path.join(result_dir, "src_dump"))

    if params['dataset'] == 'sachs':
        datasets = []
        full_data, nx_gt = cdt.data.load_dataset('sachs')
        print(full_data.shape)
        nx_gt.remove_edge('PIP2', 'PIP3')  # Fix error in cdt. Cf. the original paper
        nx_gt.add_edge('PIP3', 'PIP2')
        for _ in range(params['num_datasets']):
            data = introduce_confounders(full_data, nx_gt, params['num_hidden'], params['num_samples'])
            datasets.append((data, MixedGraph(nx_gt)))

    elif params['dataset'] == 'auto_mpg':
        datasets = []
        full_data = import_example('auto_mpg')
        print(full_data.shape)
        nx_gt = nx.DiGraph()
        nx_gt.add_edge('cylinders', 'displacement')
        nx_gt.add_edge('displacement', 'horsepower')
        nx_gt.add_edge('displacement', 'weight')
        nx_gt.add_edge('weight', 'mpg')
        # nx_gt.add_edge('horsepower', 'mpg')
        nx_gt.add_edge('origin', 'mpg')
        nx_gt.add_edge('origin', 'displacement')
        nx_gt.add_edge('weight', 'acceleration')
        nx_gt.add_edge('horsepower', 'acceleration')
        nx_gt.add_edge('horsepower', 'weight')
        nx_gt.add_edge('model_year', 'weight')

        for _ in range(params['num_datasets']):
            data = introduce_confounders(full_data, nx_gt, params['num_hidden'], params['num_samples'])
            datasets.append((data, MixedGraph(nx_gt)))

    elif params['dataset'] == 'fmri':
        datasets = []
        dataset = loadmat(os.path.join('data', 'sim2.mat'))
        adj = dataset['net'][0] != 0
        np.fill_diagonal(adj, False) # Remove self-references
        ground_truth = nx.from_numpy_array(adj, create_using=nx.DiGraph)
        ground_truth = nx.relabel_nodes(ground_truth, {i: 'V{}'.format(i) for i in range(adj.shape[0])})
        full_data = pd.DataFrame(dataset['ts'], columns=list(ground_truth.nodes))
        print(full_data.keys())
        for _ in range(params['num_datasets']):
            data = introduce_confounders(full_data, ground_truth, params['num_hidden'], params['num_samples'])
            datasets.append((data, MixedGraph(ground_truth)))

    elif params['dataset'] in ['water', 'alarm']:
        datasets = []
        for _ in range(params['num_datasets']):
            data = import_example(params['dataset'], n=params['num_samples'])
            print(data.shape)  # somehow often throws exception on first try. Just restart
            model = import_DAG(params['dataset'])
            nx_gt = nx.DiGraph()
            nx_gt.add_nodes_from(data.keys())
            for tail, head in model['model'].edges():
                nx_gt.add_edge(tail, head)
            ground_truth = MixedGraph(nx_gt)
            print('loading done')
            datasets.append((data, ground_truth))

    results = defaultdict(lambda: [])

    def worker(i, ground_truth, data):
        local_result = {}
        print(f"\n##############\nDataset {i}")
        for algo_name in params['algorithms']:
            algo = get_algo_from_name(algo_name, params)

            start_t = time.time()
            g_hat = algo.fit(data)
            end_t = time.time()
            metrics = g_hat.eval_all_metrics(ground_truth.graph)
            metrics['time'] = end_t - start_t
            #results[algo_name].append(metrics)
            print(
                f"Algorithm {algo_name}: time = {metrics['time']}s; SHD = {metrics['shd']}; skeleton f1 = {metrics['skeleton_f1']}")
            local_result[algo_name] = metrics
        return local_result


    result = Parallel(n_jobs=params['n_jobs_datasets'])(
        delayed(worker)(i, ground_truth, data) for i, (data, ground_truth) in enumerate(datasets))

    for algo_name in params['algorithms']:
        results[algo_name] += [r[algo_name] for r in result]

    for algo_name in params['algorithms']:
        df = pd.DataFrame(results[algo_name])
        df.to_csv(os.path.join(result_dir, algo_name + '.csv'))
