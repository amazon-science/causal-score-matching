# Score matching through the roof

This is the code accompanying the paper 

Montagna, F., Faller, P. M., Bloebaum, P., Kirschbaum, E., & Locatello, F. (2025). Score matching through the roof: linear, nonlinear, and latent variables causal discovery. In _Causal Learning and Reasoning (CLeaR)_. PMLR.

## Installation
For our experiments we used Python 3.12.5.
To run our code, we first need to run

    pip install -r requirements.txt

to get the dependencies.
To download the dataset from Smith et al. 2011 to the correct directory run

    cd benchmark/data
    wget https://www.fmrib.ox.ac.uk/datasets/netsim/sims.tar.gz
    tar -xzf sims.tar.gz
    mv sims/sim2.mat .
    cd ../..

## Usage

### Stand-alone AdaScore

To use our AdaScore algorithm just import the `AdaScore` class. Example usage could be

    import pandas as pd
    from causal_discovery.adascore import AdaScore 
    df = pd.read_csv('data.csv')
    algo = AdaScore(alpha_orientation=.05, alpha_confounded_leaf=.05, alpha_separations=.05)
    graph = algo.fit(df)

The result is a `networkx.DiGraph()`.
Currently, the PAG output is only supported by running FCI with the score-matching-based independence criterion from Proposition 2.
This can be used as e.g.

    
    from causal_discovery import modified_fci
    from causal_discovery.score_independence import ScoreIndependence

    cit = ScoreIndependence(df.to_numpy())
    result = fci(df.to_numpy(), independence_test_method=cit)

where `fci` has the same interface as in the `causal-learn` package, except for the possibility to set `independence_test_method` to a callable.

### Running Experiments from the paper

To run the experiments shown in the paper use e.g. the following commands

    python3 benchmark/bench_increasing_size.py --algorithms ridge nogam --num_nodes 3 5 7 --p_edge .3
    python3 benchmark/bench_increasing_samples.py --algorithms ridge nogam --num_samples 300 500 --p_edge .3
    python3 benchmark/bench_real_data.py --dataset sachs

Note, that here we indicate AdaScore with different regression functions by the name of the regression method (e.g. `ridge` or `xgboost`).
These scripts allow to run multiple algorithms at once.
Please see the respective files for available algorithms and their command line parameters.
Further, the files allow to specify certain hyperparameters of the algorithms.
Most notably, the alpha-threshold for the AdaScore algorithm.
Since most of our baselines also have an alpha parameter, the script offers to pass `alpha_others` for all algorithms
except AdaScore.

Of course, the script also offers control over the generated data.
E.g. with `num_hidden` the number of randomly selected hidden variables can be controlled and with `p_edge` the
probability of adding an edge to the Erdos-Renyi ground truth graph can be controlled.

The results are stored in the dir `logs/paper-plots` with a timestamp.
The parameters of each run are found in the `params.json` file and in the folders `data` and `graphs` are the generated
data matrices, the ground truth graph and all estimated graphs.

Single plots can be generated via

    python3 plots/plot_metrics_incr_size.py
    python3 plots/plot_metrics_incr_samples.py
    python3 plots/plot_time_incr_samples.py
    python3 plots/plot_time_incr_size.py

where the parameters allow to controll which algorithms are included in the plots and which metric should be plotted.
The parameter `--newest i` plots the `i`-th newest log folder.

To reproduce the exact plots of the paper move the respective log folder from above into a specific folder  `logs/paper-plots/incr_size/` or  `logs/paper-plots/incr_samples/` make sure this folder contains only one subfolder per parameter
setting and then use

    python3 plots/paper-plots.py

where the command line parameters of the file control which experimental setting is plotted.
The plots from Figure 2 can be generated via

    python3 plots/plot_metrics_bnlearn.py 

without manually moving folders.

## Remarks on the code structure

Overall the project is structured as follows:
The `causal_discovery` folder contains our algorithm, including an oracle-version that replaces finite-sample estimates
with graphical calculations on the ground truth graph.
The `AdaScore` class can be used as stand-alone implementation.

The `benchmark` folder contains everything required to run the benchmark, including the code that calls the `causally`
data generation, the metrics that we use and the benchmarking scripts themselves.
It is worth noting that we added a `utils/algo_wrapper.py` file to unify the interface of algorithms from different
packages.
The classes in `utils/causal_graphs.py` allow us to dynamically call the correct metrics (for different graph types)
without `if-else` clauses in the main scripts.

`plots` contains the respective scripts to plot the data and `tests` contains unit tests for our code.



