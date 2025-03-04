# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

import argparse
import glob
import json
import os
import sys
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# For the legend
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

sns.set_style('whitegrid')


def generate_p_value_table(df, x_feature, reference_algo, algorithms, alternative: str = 'less'):
    """
    Generates a table of Mann-Whitney U test p-values comparing the reference algorithm
    against all other algorithms for each num_nodes.
    """
    # Get unique num_nodes and algorithms
    feature_values = sorted(df[x_feature].unique())

    # Prepare a DataFrame to store p-values, with algorithms as rows and num_nodes as columns
    p_value_table = pd.DataFrame(index=[algo for algo in algorithms if algo != reference_algo],
                                 columns=feature_values
                                 )

    # Loop through each num_nodes value
    for value in feature_values:
        # Filter the dataframe for the current num_nodes
        filtered_df = df[df[x_feature] == value]

        # Get SHD values for the reference algorithm
        shd_reference = filtered_df[filtered_df['algo'] == reference_algo]['shd']
        # print(reference_algo, shapiro(shd_reference).pvalue, shapiro(shd_reference).pvalue < .05)

        # Loop through each algorithm (except the reference)
        for algo in p_value_table.index:
            # Get SHD values for the current algorithm
            shd_algo = filtered_df[filtered_df['algo'] == algo]['shd']

            # Perform Mann-Whitney U test
            print(shd_reference.shape, shd_algo.shape, list(df['algo'].unique()))
            _, p_value = mannwhitneyu(shd_reference, shd_algo, alternative=alternative, method='exact')
            # _, p_value = ttest_ind(shd_reference, shd_algo, alternative=alternative)

            # Store the p-value in the corresponding cell
            p_value_table.loc[algo, value] = p_value

    return p_value_table

def generate_mannwhitneyu_table(df, x_feature, reference_algo, algorithms, caption):
    df_less = generate_p_value_table(df, x_feature, reference_algo, algorithms, 'less')
    df_greater = generate_p_value_table(df, x_feature, reference_algo, algorithms, 'greater')
    # Create MultiIndex for columns with settings as level 0 and node counts as level 1
    df_less.columns = pd.MultiIndex.from_product([['less'], df_less.columns])
    df_greater.columns = pd.MultiIndex.from_product([['greater'], df_greater])

    # Concatenate along columns
    df_combined = pd.concat([df_less, df_greater], axis=1)

    # Export to LaTeX
    return df_combined.to_latex(multicolumn=True, multirow=True, caption=caption, float_format="{:0.5f}".format)



def custom_palette(n_colors):
    palette = [
                  "#D4AC0D",  # yellow
                  "#953553",  # red
                  "#3498DB",  # blue
                  "#7DCEA0",  # green
                  "#FA8072",  # salmon
                  "#9384C0",  # purple
              ][:n_colors]

    return sns.color_palette(palette, len(palette))


def make_ax_violinplot(
        seaborn_df: List,
        ax: plt.Axes,
        metric: str,
        colors: List[str],
        title: str = None,
        x: str = "scenario",
):
    """Make violinplot on specified axes subplot.

    Make pandas DataFrame from records and graph violinplot at axes[row, col]. 
    """
    sns_ax = sns.boxplot(
        ax=ax, data=seaborn_df, x=x, y=metric, hue="algo", palette=colors, boxprops=dict(alpha=.5), width=0.7
    )

    if title is not None:
        sns_ax.set_title(title)

    sns_ax.spines["bottom"].set_color('black')
    sns.despine(left=True)  # remove left border
    if metric == 'time':
        sns_ax.set_yscale('log')
    else:
        sns_ax.set_ylim(bottom=-0)

    return sns_ax


def make_titles(params_path):
    with open(params_path) as f:
        params = json.load(f)

    if 'non-additive' == params['scm']:
        if params['p_edge'] == .5:
            axtitle = 'dense'
        elif params['p_edge'] == .3:
            axtitle = 'sparse'
    else:
        axtitle = params["scm"]
    if params["num_hidden"] > 0:
        figtitle = "Latent variables model"
    else:
        figtitle = "Fully observable model"
    return axtitle, figtitle


def make_plot_name(params_path):
    with open(params_path) as f:
        params = json.load(f)
    pdfname = []
    if params["num_hidden"] > 0:
        pdfname.append("latent")
    else:
        pdfname.append("observable")

    if params["p_edge"] == 0.3:
        pdfname.append("sparse")
    elif params["p_edge"] == 0.5:
        pdfname.append("dense")

    return "_".join(pdfname)


if __name__ == "__main__":

    # Command linear arguments
    parser = argparse.ArgumentParser(description='Experiment with increasing node size.')
    parser.add_argument('--algorithms', nargs='+', default=['ridge', 'camuv', 'nogam', 'rcd', 'lingam', 'random'])
    parser.add_argument('--metric', default='shd')
    parser.add_argument('--x_feature', default='num_nodes')
    parser.add_argument('--scms', nargs='+', default=['linear', 'nonlinear'])  # scms in the plot
    parser.add_argument('--p_edge', default=0.5, type=float)  # sparsity of the plot
    parser.add_argument('--num_hidden', default=2, type=int)  # confounded/not plot
    params = vars(parser.parse_args())

    algorithms = params["algorithms"]
    metric = params["metric"]

    if params['x_feature'] == 'num_nodes':
        log_name = 'incr_size'
        x_label = 'number of nodes'
    elif params['x_feature'] == 'num_samples':
        log_name = 'incr_samples'
        x_label = 'number of samples'
    else:
        raise ValueError()

    # Get the dirs with data for the plots
    logs_dirs = glob.glob(os.path.join('benchmark', 'logs', 'paper-plots', log_name, '*'))
    output_dir = os.path.join('benchmark', 'logs', 'paper-plots')
    plots_dirs = list()
    reference_params_path = dict()
    for dir in logs_dirs:
        if os.path.isdir(dir):
            log_params_path = os.path.join(dir, "params.json")
            with open(log_params_path) as f:
                log_params = json.load(f)
            if (  # Directories with FCI make a mess out of this. TODO: add fci csv files inside the right dirs
                    log_params["scm"] in params["scms"] and \
                    log_params["num_hidden"] == params["num_hidden"] and \
                    log_params["p_edge"] == params["p_edge"]
            ):
                plots_dirs.append(dir)
                reference_params_path = log_params_path

    if len(plots_dirs) == 0:
        raise ValueError("There are no data matching your requirements. Aborting plot")

    log_params_path = os.path.join(plots_dirs[0], "params.json")
    with open(log_params_path) as f:
        log_params = json.load(f)
        if log_params['scm'] != params['scms'][0]:
            plots_dirs = plots_dirs[::-1]  # Switch order, if it doesn't match given one

    pdf_name = make_plot_name(reference_params_path) + ".pdf"
    table_name = make_plot_name(reference_params_path) + ".tex"
    table_string = ''

    # Create axes
    fig, axes = plt.subplots(1, len(plots_dirs), figsize=(12 * len(plots_dirs), 8))
    for i in range(len(plots_dirs)):
        file_dir = plots_dirs[i]
        if len(plots_dirs) > 1:
            ax = axes[i]
        else:
            ax = axes

        # Title and pdfname
        params_path = os.path.join(file_dir, "params.json")
        axtitle, figtitle = make_titles(params_path)

        # Combined logs in df
        dfs = []
        for algo in algorithms:
            file = algo + '.csv'
            df = pd.read_csv(os.path.join(file_dir, file), index_col=0)
            df['algo'] = algo  # Add a column to identify the source file
            dfs.append(df)
        combined_df = pd.concat(dfs)

        table_string += generate_mannwhitneyu_table(combined_df, params['x_feature'], 'ridge', params['algorithms'],
                                                    'p-values for stochastic ordering: ' + axtitle.lower() + ' ' + figtitle) + '\n\n'

        # Boxplot
        colors = custom_palette(len(algorithms))
        ax = make_ax_violinplot(combined_df, ax, metric, colors, title=axtitle, x=params['x_feature'])

        # labels
        if metric == 'bi_edge_f1':
            ax.set_ylabel(r'$F_1$ score: undirected edges', size=32)
        elif metric == 'direct_edge_f1':
            ax.set_ylabel(r'$F_1$ score: directed edges', size=32)
        elif metric == 'skeleton_f1':
            ax.set_ylabel(r'$F_1$ score: skeleton', size=32)
        elif metric == 'time':
            ax.set_ylabel('time [s]', size=32)
        else:
            ax.set_ylabel(metric, size=32)
        ax.set_xlabel(x_label, size=32)

        # ticks (increase xticks spacing)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=28)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=28)

        # Ax title
        ax.set_title(ax.get_title(), fontsize=32)
        ax.get_legend().remove()

    # title = fig.suptitle(figtitle, fontsize=34)
    # title.set_position([0.5, 0.98])
    fig.tight_layout(h_pad=5, w_pad=3)  # h_pad add space between rows, w_pad between cols
    fig.subplots_adjust(top=.85)  # top add space above the first row
    plt.savefig(os.path.join(output_dir, f'{metric}_{pdf_name}'))
    with open(os.path.join(output_dir, f'table_{metric}_{table_name}'), 'w') as file:
        file.write(table_string)
    plt.close("all")

    # Make legend
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    lines = legend = [Line2D([0], [0], color=c, lw=4) for c in colors]
    labels = [algo if algo not in ["gam", 'ridge', 'score_fci'] else "adascore" for algo in algorithms]  # map names to adascore
    ax.clear()  # Remove the axis data and labels
    legend = ax.legend(lines, labels, ncol=len(labels), loc='center', fontsize=17, borderaxespad=0)
    ax.axis('off')  # Hide axes
    # Set alpha to legend lines
    for lh in legend.legend_handles:
        lh.set_alpha(.7)

    fig.savefig((os.path.join(output_dir, f'legend.pdf')))
    plt.close("all")
