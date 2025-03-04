# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List

import networkx as nx
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmark.data.generate_data import get_confounded_datasets, get_pag_skel_with_ada_orientations
from benchmark.utils.custom_causally import NonAdditiveNoiseModel, AdditivePolynomialMechanism, CAMUVMechanism, \
    AdditiveNeuralNetMechanism
from benchmark.utils.causal_graphs import MixedGraph
from benchmark.utils.algo_wrappers import get_algo_from_name
from benchmark.utils.cache_source_files import copy_referenced_files_to

# Causally imports
import causally.scm.scm as scm
import causally.graph.random_graph as rg
import causally.scm.noise as noise
import causally.scm.causal_mechanism as cm
from causally.scm.context import SCMContext


def get_causally_datasets(
        num_datasets: int,
        num_samples: int,
        num_observed_nodes: int,
        num_hidden: int,
        scm_type: str,
        noise_dist: str,
        expected_degree: int = None,
        p_edge=None,
        standardize=True,
        seed: int = None
) -> List[Tuple[pd.DataFrame, nx.DiGraph]]:
    if p_edge is not None and expected_degree is not None:
        raise ValueError("Can not set both p_edge and m_edge. Only one")
    if p_edge is None and expected_degree is None:
        raise ValueError("You must explicitly set one value between p_edge and m_edge")

    datasets = []
    while(len(datasets) < num_datasets):
        # Erdos-Renyi graph generator
        num_nodes = num_observed_nodes + num_hidden
        graph_generator = rg.ErdosRenyi(
            num_nodes=num_nodes, expected_degree=expected_degree, p_edge=p_edge, min_num_edges=math.ceil(num_nodes / 2)
        )

        # Generator of the noise terms

        if noise_dist == "gauss":
            noise_generator = noise.Normal()
        elif noise_dist == "mlp":
            noise_generator = noise.MLPNoise(a_weight=-1.5, b_weight=1.5)
        elif noise_dist == "uniform":
            noise_generator = noise.Uniform(-2, 2)  # Std ~ 1

        # Structural causal model
        context = SCMContext()  # context for assumptions
        if scm_type == "nonlinear":
            causal_mechanism = cm.NeuralNetMechanism()
            model = scm.AdditiveNoiseModel(
                num_samples=num_samples,
                graph_generator=graph_generator,
                noise_generator=noise_generator,
                causal_mechanism=causal_mechanism,
                scm_context=context,
                seed=seed
            )
        elif scm_type == "non-additive":
            causal_mechanism = cm.NeuralNetMechanism()
            model = NonAdditiveNoiseModel(
                num_samples=num_samples,
                graph_generator=graph_generator,
                noise_generator=noise_generator,
                causal_mechanism=causal_mechanism,
                scm_context=context,
                seed=seed
            )
        elif scm_type == "linear":
            if noise == "gauss":
                raise ValueError("Can not have gaussian noise for linear mechanisms.")
            model = scm.LinearModel(
                num_samples=num_samples,
                graph_generator=graph_generator,
                noise_generator=noise_generator,
                scm_context=context,
                seed=seed,
                max_weight=3,
                min_weight=-3,
                min_abs_weight=.5,
            )
        elif scm_type == "poly_additive":
            causal_mechanism = AdditivePolynomialMechanism()
            model = scm.AdditiveNoiseModel(
                num_samples=num_samples,
                graph_generator=graph_generator,
                noise_generator=noise_generator,
                causal_mechanism=causal_mechanism,
                scm_context=context,
                seed=seed
            )
        elif scm_type == "camuv_paper":
            causal_mechanism = CAMUVMechanism()
            model = scm.AdditiveNoiseModel(
                num_samples=num_samples,
                graph_generator=graph_generator,
                noise_generator=noise_generator,
                causal_mechanism=causal_mechanism,
                scm_context=context,
                seed=seed
            )

        elif scm_type == "nn_additive":
            causal_mechanism = AdditiveNeuralNetMechanism()
            model = scm.AdditiveNoiseModel(
                num_samples=num_samples,
                graph_generator=graph_generator,
                noise_generator=noise_generator,
                causal_mechanism=causal_mechanism,
                scm_context=context,
                seed=seed
            )

        X, y = model.sample()
        if standardize:
            marginal_std = np.std(X, axis=0)
            for i in range(len(marginal_std)):
                X[:, i] = X[:, i] / marginal_std[i]

        # Change variables name and create nx gt
        nodes_map = {node: f"V{node + 1}" for node in range(0, num_nodes)}
        gt = nx.from_numpy_array(y, create_using=nx.DiGraph)
        gt = nx.relabel_nodes(gt, nodes_map)
        data = pd.DataFrame(X).rename(columns=nodes_map)

        # Drop hidden variables
        if num_hidden > 0:
            # Pick number at random
            hidden_idxs = np.random.choice(np.array(range(0, num_nodes)), size=(num_hidden,), replace=False)
            data = data.drop([nodes_map[k] for k in hidden_idxs], axis=1)
            indicate_ucp = scm != 'linear'
            marginal_gt = get_pag_skel_with_ada_orientations(gt, data.keys(), indicate_ucp)
            num_bi_edges = len([(x, y) for x in marginal_gt.nodes for y in marginal_gt.nodes
                                if marginal_gt.has_edge(x, y) and marginal_gt.has_edge(y, x)])

            if num_bi_edges == 0:
                print("No bidirected edges. Regenerate.")
                continue

        datasets.append((data, gt))

    return datasets


if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Experiment with increasing node size.')
    parser.add_argument('--algorithms', nargs='+', default=['ridge', 'camuv', 'nogam', 'rcd', 'lingam', 'random', 'fully_random'])
    parser.add_argument('--num_nodes', nargs='+', default=[3, 5, 7, 9])
    parser.add_argument('--num_hidden', default=2, type=int)
    parser.add_argument('--num_samples', default=1000, type=int)
    parser.add_argument('--num_datasets', default=10, type=int)
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

    result_dir = os.path.join('benchmark', 'logs', 'paper-plots', 'incr_size_') + time.strftime('%y.%m.%d_%H.%M.%S')
    Path(result_dir).mkdir(parents=True, exist_ok=False)
    with open(os.path.join(result_dir, 'params.json'), 'w') as file:
        json.dump(params, file)
    copy_referenced_files_to(__file__, os.path.join(result_dir, "src_dump"))

    results = defaultdict(lambda: [])
    for num_nodes in params['num_nodes']:
        graph_dir = os.path.join(result_dir, 'graphs', 'num_nodes_{}'.format(num_nodes))
        data_dir = os.path.join(result_dir, 'data', 'num_nodes_{}'.format(num_nodes))
        Path(graph_dir).mkdir(parents=True, exist_ok=False)
        Path(data_dir).mkdir(parents=True, exist_ok=False)
        if params['scm'] != 'ram':
            datasets = get_causally_datasets(
                    num_datasets=params['num_datasets'],
                    num_samples=params['num_samples'],
                    num_observed_nodes=int(num_nodes),
                    num_hidden=params["num_hidden"],
                    scm_type=params["scm"],
                    noise_dist=params["noise"],
                    p_edge=params["p_edge"],
                    expected_degree=params["m_edge"],
                    standardize=True
                )
        else:
            datasets = get_confounded_datasets(params['num_datasets'],int(num_nodes), params['num_hidden'],
                                               params['num_samples'], params['p_edge'], params['scm'])


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
                metrics['num_nodes'] = num_nodes
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
