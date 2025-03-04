# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

import argparse
import glob
import json
import os
import sys

import pandas as pd
from matplotlib import pyplot as plt
from numpy import sort

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmark.utils.cache_source_files import copy_referenced_files_to

palette = [
    "#D4AC0D",  # yellow
    "#953553",  # red
    "#3498DB",  # blue
    "#7DCEA0",  # green
    "#FA8072",  # salmon
    "#9384C0",  # purple
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment with increasing node size.')
    parser.add_argument('--algorithms', nargs='+', default=['ridge', 'camuv', 'nogam', 'rcd', 'lingam'])
    parser.add_argument('--newest', default=1, type=int)
    parser.add_argument('--metric', default='shd')
    params = vars(parser.parse_args())

    file_dir = sort(glob.glob(os.path.join('benchmark', 'logs', 'paper-plots', 'real_data*')))[-params['newest']]
    print(file_dir)
    result_dir = os.path.join('.', 'img', 'incr_size')
    result_dir = file_dir
    copy_referenced_files_to(__file__, os.path.join(result_dir, params['metric'] + "_real_data_src_dump/"))

    with open(os.path.join(file_dir, 'params.json')) as file:
        experiment_params = json.load(file)

    metrics = {}
    for algo in params['algorithms']:
        file = algo + '.csv'
        df = pd.read_csv(os.path.join(file_dir, file), index_col=0)
        metrics[algo] = df[params['metric']].iloc[0]

    # Plotting
    fig, ax = plt.subplots()
    positions = {}  # Store positions for each group
    legend_artists = {}
    offset = -.2
    for algo in params['algorithms']:
        plt.bar(list(range(len(metrics))), list(metrics.values()), color=palette[:len(metrics.keys())])
        plt.xticks(list(range(len(metrics))), [m if m not in  ['ridge', 'scam'] else 'AdaScore' for m in metrics.keys()])
    # ax.set_title('Boxplots of SHD by Num Nodes')
    ax.set_ylabel(params['metric'])
    plt.title(experiment_params['dataset'] + ' dataset')

    plt.savefig(os.path.join(result_dir, params['metric'] + '_' + experiment_params['dataset'] + '.pdf'))
    plt.show()
