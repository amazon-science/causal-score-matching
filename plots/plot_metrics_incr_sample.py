# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

import argparse
import glob
import os
import sys

import pandas as pd
from matplotlib import pyplot as plt
from numpy import sort

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmark.utils.cache_source_files import copy_referenced_files_to

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment with increasing node size.')
    parser.add_argument('--algorithms', nargs='+', default=['ridge', 'camuv'])
    parser.add_argument('--newest', default=1, type=int)
    parser.add_argument('--metric', default='shd')
    params = vars(parser.parse_args())

    file_dir = sort(glob.glob(os.path.join('benchmark', 'logs', 'paper-plots', 'incr_samples_*')))[-params['newest']]
    result_dir = os.path.join('.', 'img', 'incr_samples')
    copy_referenced_files_to(__file__, os.path.join(result_dir, params['metric'] + "_incr_sample_src_dump/"))

    dfs = []
    for algo in params['algorithms']:
        file = algo + '.csv'
        df = pd.read_csv(os.path.join(file_dir, file), index_col=0)
        df['algo'] = algo  # Add a column to identify the source file
        dfs.append(df)
    combined_df = pd.concat(dfs)

    # Plotting
    fig, ax = plt.subplots()
    positions = {}  # Store positions for each group
    legend_artists = {}
    offset = -.2
    colour_pallet = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for algo, face_colour, median_colour in zip(params['algorithm'], colour_pallet, colour_pallet[1:]):
        group = combined_df.groupby('algo').get_group(algo)
        gb_df = group.groupby('num_samples')[params['metric']]
        for num_nodes, values in gb_df:
            if num_nodes not in positions:
                positions[num_nodes] = len(positions)
            bp = ax.boxplot(values,
                            positions=[positions[num_nodes] + offset],
                            patch_artist=True,
                            boxprops=dict(facecolor=face_colour),
                            medianprops=dict(color=median_colour)
                            )
            legend_artists[algo] = bp['boxes'][0]
        offset += .4 / ((len(params['algorithms']) - 1) if len(params['algorithms']) > 1 else 1)
    # ax.set_title('Boxplots of SHD by Num Nodes')
    ax.set_ylabel(params['metric'])
    ax.set_xlabel('Num Nodes')
    ax.legend(legend_artists.values(), legend_artists.keys())  # Add legend

    # Adjust x-axis ticks
    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels(positions.keys())

    plt.savefig(os.path.join(result_dir, params['metric'] + '_incr_size.pdf'))
    plt.show()
