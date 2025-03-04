# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from typing import Union, Tuple, List
from causally.scm.causal_mechanism import PredictionModel, CustomMechanism
from causally.scm.scm import BaseStructuralCausalModel
from causally.scm.noise import RandomNoiseDistribution, Distribution
from causally.scm.context import SCMContext
from causally.graph.random_graph import GraphGenerator
from numpy._typing import NDArray

from benchmark.data.generator import NNMechanism


class NonAdditiveNoiseModel(BaseStructuralCausalModel):
    """Class for data generation from a nonlinear model with non-additive noise.

    Parameters
    ----------
    num_samples: int
        Number of samples in the dataset.
    graph_generator: GraphGenerator
        Random graph generator implementing the ``get_random_graph`` method.
    noise_generator:  Distribution
        Sampler of the noise random variables. It must be an instance of
        a class inheriting from ``causally.scm.noise.Distribution``, implementing
        the ``sample`` method.
    causal_mechanism: PredictionModel
        Object for the generation of the nonlinear causal mechanism.
        It must be an instance of a class inheriting from
        ``causally.scm.causal_mechanism.PredictionModel``, implementing
        the ``predict`` method.
    scm_context: SCMContext, default None
        ``SCMContext`` object specifying the modeling assumptions of the SCM.
        If ``None`` this is equivalent to an ``SCMContext`` object with no
        assumption specified.
    seed: int, default None
        Seed for reproducibility. If ``None``, then the random seed is not set.
    """

    def __init__(
        self,
        num_samples: int,
        graph_generator: GraphGenerator,
        noise_generator: Union[RandomNoiseDistribution, Distribution],
        causal_mechanism: PredictionModel,
        scm_context: SCMContext = None,
        seed: int = None,
    ):
        super().__init__(
            num_samples, graph_generator, noise_generator, scm_context, seed
        )
        self.causal_mechanism = causal_mechanism

    def _sample_mechanism(self, parents: np.array, child_noise: np.array) -> np.array:
        effect = self.causal_mechanism.predict(np.concatenate([parents, child_noise[:, None]], axis=1))
        return effect


class AdditivePolynomialMechanism(CustomMechanism):
    def __init__(self, max_degree: int = 5, coef_range: Tuple[float, float] = (-2, 2)):
        self.coef_range = coef_range
        self.max_degree = max_degree

    def predict(self, X: np.array) -> np.array:
        mechanisms = []
        for pa_idx in range(X.shape[1]):
            coefs = np.random.uniform(low=self.coef_range[0], high=self.coef_range[1], size=self.max_degree)
            mechanisms.append(lambda pa: np.sum([coefs[i] * pa ** i for i in range(self.max_degree)], axis=0)[:, 0])

        output = np.zeros(X.shape[0])
        for i in range(X.shape[1]):
            output += mechanisms[i](np.expand_dims(X[:, i], -1))
        return output


class CAMUVMechanism(CustomMechanism):
    def __init__(self, offset_range: Tuple[float, float] = (-5, 5),
                 intercept_range: Tuple[float, float] = (-1, 1),
                 exponents: List[int] = (2, 3)):
        self.offset_range = offset_range
        self.intercept_range = intercept_range
        self.exponents = exponents

    def predict(self, X: np.array) -> np.array:
        mechanisms = []
        for pa_idx in range(X.shape[1]):
            offset = np.random.uniform(low=self.offset_range[0], high=self.offset_range[1])
            intercept = np.random.uniform(low=self.offset_range[0], high=self.offset_range[1])
            expo = np.random.choice(self.exponents)

            mechanisms.append(lambda pa: (np.power(pa + offset, expo) + intercept))

        output = np.zeros(X.shape[0])
        for i in range(X.shape[1]):
            output += mechanisms[i](X[:, i])
        return output


class AdditiveNeuralNetMechanism(CustomMechanism):
    def __init__(self, num_hidden: int = 10, coef_range: Tuple[float, float] = (-3, 3)):
        self.num_hidden = num_hidden
        self.coef_range = coef_range


    def predict(self, X: NDArray) -> NDArray:
        mechanisms = []
        for _ in range(X.shape[1]):
            mechanisms.append(NNMechanism(1, self.num_hidden, self.coef_range))

        output = np.zeros(X.shape[0])
        for i in range(X.shape[1]):
            output += mechanisms[i](np.expand_dims(X[:, i], -1))
        return output