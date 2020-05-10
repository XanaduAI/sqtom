# Copyright 2019-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Basic tests for the functions in fitting_1d"""

import pytest
import numpy as np
from sqtom.forward_solver import degenerate_pmf
from sqtom.fitting_1d import fit_1d, threshold_1d, marginal_calcs_1d


@pytest.mark.parametrize("do_not_vary", ["eta", "noise", []])
@pytest.mark.parametrize("n_modes", [1, 2, 3])
@pytest.mark.parametrize("threshold", [False, 10])
def test_exact_model_1d(n_modes, threshold, do_not_vary):
    """Test that the fitting is correct when the guess is exactly the correct answer"""
    sq_n = 0.7 * (0.5 ** np.arange(n_modes))
    noise = 0.1
    eta = 0.7
    params = {"sq_" + str(i): sq_n[i] for i in range(n_modes)}
    params["n_modes"] = n_modes
    params["eta"] = eta
    params["noise"] = noise
    if threshold is not False:
        probs = threshold_1d(degenerate_pmf(params), threshold)
    else:
        probs = degenerate_pmf(params)
    fit = fit_1d(probs, params, threshold=threshold, do_not_vary=do_not_vary)
    assert np.allclose(fit.chisqr, 0.0)


def test_marginal_calcs_1d():
    """Tests that matginal_calcs_1d returns the correct values as an array"""
    nmean = 1.0
    ps = degenerate_pmf({"n_modes": 1.0, "sq_0": nmean})
    assert np.allclose(
        marginal_calcs_1d(ps, as_dict=False), np.array([nmean, 3 + 1 / nmean])
    )
