# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from causallearn.utils.cit import CIT_Base, NO_SPECIFIED_PARAMETERS_MSG
from dodiscover.toporder._base import SteinMixin
from scipy.stats import ttest_1samp


class ScoreIndependence(CIT_Base):
    def __init__(self, data, eta_g: float = 0.001, eta_h: float = 0.001, cache_hessian: bool = True, **kwargs):
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent('score_independence', NO_SPECIFIED_PARAMETERS_MSG)
        self.assert_input_data_is_valid()
        self.stein_estimator = SteinMixin()
        self.eta_g = eta_g
        self.eta_h = eta_h
        self.cache_hessian = cache_hessian
        self.hessian_cache = {}

    def __call__(self, X, Y, condition_set=None):
        '''
        Perform an independence 'test' using non-diagonal entries of the hessian of log density.

        Parameters
        ----------
        X, Y and condition_set : column indices of data

        Returns
        -------
        p : the p-value of the test
        '''
        Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if cache_key in self.pvalue_cache: return self.pvalue_cache[cache_key]
        subset = np.sort(list(set(Xs).union(set(Ys).union(set(condition_set)))))
        hessian_key = '.'.join(map(str, subset))
        if hessian_key not in self.hessian_cache:
            hessian = self.stein_estimator.hessian(self.data[:, subset], self.eta_g, self.eta_h)
            if self.cache_hessian:
                self.hessian_cache[hessian_key] = hessian
        else:
            hessian = self.hessian_cache[hessian_key]

        new_idx = {old: new for new, old in enumerate(subset)}
        p_values = []
        for x in Xs:
            for y in Ys:
                p_values.append(ttest_1samp(hessian[:, new_idx[x], new_idx[y]], 0).pvalue)
        min_p = np.min(p_values)  # dependant if any x, y in Xs, Ys are dependant
        self.pvalue_cache[cache_key] = min_p
        return min_p
