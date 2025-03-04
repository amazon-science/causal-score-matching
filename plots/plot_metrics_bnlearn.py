# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

import argparse
import glob
import json
import os
import sys

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from numpy import sort

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plots.paper_plots import custom_palette, make_ax_violinplot

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
    matplotlib.rcParams.update({'font.size': 20})
    parser = argparse.ArgumentParser(description='Experiment with increasing node size.')
    parser.add_argument('--algorithms', nargs='+', default=['ridge', 'camuv', 'nogam', 'rcd', 'lingam'])
    parser.add_argument('--newest', default=1, type=int)
    parser.add_argument('--metric', default='shd')
    params = vars(parser.parse_args())

    file_dir = sort(glob.glob(os.path.join('benchmark', 'logs', 'paper-plots', 'real_data*')))[-params['newest']]
    print(file_dir)
    result_dir = os.path.join('benchmark', 'logs', 'paper-plots', 'incr_size')
    copy_referenced_files_to(__file__, os.path.join(result_dir, params['metric'] + "_real_data_src_dump/"))

    with open(os.path.join(file_dir, 'params.json')) as file:
        experiment_params = json.load(file)

    dfs = []
    for algo in params['algorithms']:
        file = algo + '.csv'
        df = pd.read_csv(os.path.join(file_dir, file), index_col=0)
        df['algorithm'] = (algo  if algo not in ['ridge'] else 'adascore') # Add a column to identify the source file
        dfs.append(df)
    combined_df = pd.concat(dfs)

    # Plotting
    fig, ax = plt.subplots()
    positions = {}  # Store positions for each group
    legend_artists = {}
    offset = -.2
    # Boxplot
    colors = custom_palette(len(params['algorithms']))
    title = os.path.basename(os.path.normpath(file_dir))
    ax = make_ax_violinplot(combined_df, ax, params['metric'], colors, title="", x="algorithm")
    # ax.set_title('Boxplots of SHD by Num Nodes')
    ax.set_ylabel(params['metric'])
    #plt.title(experiment_params['dataset'] + ' dataset')
    plt.tight_layout()

    plt.savefig(os.path.join(result_dir, params['metric'] + '_' + experiment_params['dataset'] + '.pdf'))
    plt.show()
