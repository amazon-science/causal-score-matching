# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import warnings
from typing import List, Tuple, Callable

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.Graph import Graph
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ConstraintBased.FCI import reorientAllWith, \
    rule0, \
    removeByPossibleDsep, \
    rulesR1R2cycle, \
    ruleR3, \
    ruleR4B, get_color_edges
from causallearn.utils.FAS import fas
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.cit import *
from numpy import ndarray


def fci(dataset: ndarray,
        independence_test_method: str | Callable = fisherz,
        alpha: float = 0.05,
        depth: int = -1,
        max_path_length: int = -1,
        verbose: bool = False,
        background_knowledge: BackgroundKnowledge | None = None,
        show_progress: bool = True,
        **kwargs) -> Tuple[Graph, List[Edge]]:
    """
    Perform Fast Causal Inference (FCI) algorithm for causal discovery

    Parameters
    ----------
    dataset: data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of
            samples and n_features is the number of features.
    independence_test_method: str, name of the function of the independence test being used
            [fisherz, chisq, gsq, kci]
           - fisherz: Fisher's Z conditional independence test
           - chisq: Chi-squared conditional independence test
           - gsq: G-squared conditional independence test
           - kci: Kernel-based conditional independence test
    alpha: float, desired significance level of independence tests (p_value) in (0,1)
    depth: The depth for the fast adjacency search, or -1 if unlimited
    max_path_length: the maximum length of any discriminating path, or -1 if unlimited.
    verbose: True is verbose output should be printed or logged
    background_knowledge: background knowledge

    Returns
    -------
    graph : a GeneralGraph object, where graph.graph[j,i]=1 and graph.graph[i,j]=-1 indicates  i --> j ,
                    graph.graph[i,j] = graph.graph[j,i] = -1 indicates i --- j,
                    graph.graph[i,j] = graph.graph[j,i] = 1 indicates i <-> j,
                    graph.graph[j,i]=1 and graph.graph[i,j]=2 indicates  i o-> j.
    edges : list
        Contains graph's edges properties.
        If edge.properties have the Property 'nl', then there is no latent confounder. Otherwise,
            there are possibly latent confounders.
        If edge.properties have the Property 'dd', then it is definitely direct. Otherwise,
            it is possibly direct.
        If edge.properties have the Property 'pl', then there are possibly latent confounders. Otherwise,
            there is no latent confounder.
        If edge.properties have the Property 'pd', then it is possibly direct. Otherwise,
            it is definitely direct.
    """

    if dataset.shape[0] < dataset.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")

    if isinstance(independence_test_method, str):
        independence_test_method = CIT(dataset, method=independence_test_method, **kwargs)
    elif not callable(independence_test_method):
        raise TypeError("independence_test_method must be 'str' or 'Callable! Not {}".format(type(
            independence_test_method
        )
        )
        )

    ## ------- check parameters ------------
    if (depth is None) or type(depth) != int:
        raise TypeError("'depth' must be 'int' type!")
    if (background_knowledge is not None) and type(background_knowledge) != BackgroundKnowledge:
        raise TypeError("'background_knowledge' must be 'BackgroundKnowledge' type!")
    if type(max_path_length) != int:
        raise TypeError("'max_path_length' must be 'int' type!")
    ## ------- end check parameters ------------

    nodes = []
    for i in range(dataset.shape[1]):
        node = GraphNode(f"X{i + 1}")
        node.add_attribute("id", i)
        nodes.append(node)

    # FAS (“Fast Adjacency Search”) is the adjacency search of the PC algorithm, used as a first step for the FCI algorithm.
    graph, sep_sets, test_results = fas(dataset,
                                        nodes,
                                        independence_test_method=independence_test_method,
                                        alpha=alpha,
                                        knowledge=background_knowledge,
                                        depth=depth,
                                        verbose=verbose,
                                        show_progress=show_progress
                                        )

    reorientAllWith(graph, Endpoint.CIRCLE)

    rule0(graph, nodes, sep_sets, background_knowledge, verbose)

    removeByPossibleDsep(graph, independence_test_method, alpha, sep_sets)

    reorientAllWith(graph, Endpoint.CIRCLE)
    rule0(graph, nodes, sep_sets, background_knowledge, verbose)

    change_flag = True
    first_time = True

    while change_flag:
        change_flag = False
        change_flag = rulesR1R2cycle(graph, background_knowledge, change_flag, verbose)
        change_flag = ruleR3(graph, sep_sets, background_knowledge, change_flag, verbose)

        if change_flag or (first_time and background_knowledge is not None and
                           len(background_knowledge.forbidden_rules_specs) > 0 and
                           len(background_knowledge.required_rules_specs) > 0 and
                           len(background_knowledge.tier_map.keys()) > 0):
            change_flag = ruleR4B(graph, max_path_length, dataset, independence_test_method, alpha, sep_sets,
                                  change_flag,
                                  background_knowledge, verbose
                                  )

            first_time = False

            if verbose:
                print("Epoch")

    graph.set_pag(True)

    edges = get_color_edges(graph)

    return graph, edges
