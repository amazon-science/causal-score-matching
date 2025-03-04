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
    parser.add_argument('--algorithms', nargs='+', default=['ridge', 'camuv', 'nogam', 'rcd', 'lingam', 'random', 'fully_random'])
    parser.add_argument('--newest', default=1, type=int)
    params = vars(parser.parse_args())

    file_dir = sort(glob.glob(os.path.join('benchmark', 'logs', 'paper-plots', 'incr_size_*')))[-params['newest']]
    result_dir = os.path.join('.', 'img', 'incr_size')
    copy_referenced_files_to(__file__, os.path.join(result_dir, "runtimes_src_dump"))

    _, ax = plt.subplots()
    for algo in params['algorithms']:
        file = algo + '.csv'
        df = pd.read_csv(os.path.join(file_dir, file), index_col=0)
        gb_time = df.groupby('num_nodes')['time']
        ax.errorbar(gb_time.indices.keys(), gb_time.mean(), yerr=gb_time.std(), label=file)
    ax.set_ylabel(r'Runtime [s]')
    ax.set_xlabel(r'Num nodes')
    ax.legend()
    plt.savefig(os.path.join(result_dir, 'runtimes.pdf'))
    plt.show()
