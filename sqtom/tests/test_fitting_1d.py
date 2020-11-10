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
from thewalrus.samples import photon_number_sampler
from sqtom.forward_solver import degenerate_pmf
from sqtom.fitting_1d import (
    two_schmidt_mode_guess,
    marginal_calcs_1d,
    gen_hist_1d,
    threshold_1d,
    fit_1d,
)


@pytest.mark.parametrize("sq_0", [0.1, 1.0])
@pytest.mark.parametrize("eta", [0.3, 1.0])
def test_gen_hist_1d(sq_0, eta):
    """Check that a histogram is constructed correctly for a degenerate squeezing source"""
    nsamples = 100_000
    nmax = 30
    pmf_init = degenerate_pmf({"sq_0": sq_0, "n_modes": 1, "eta": eta}, cutoff=nmax)
    samples = photon_number_sampler(pmf_init, nsamples)
    pmf_gen = gen_hist_1d(samples)
    pmf_final = degenerate_pmf({"sq_0": sq_0, "n_modes": 1, "eta": eta}, cutoff=np.max(samples))
    atol = 10 / np.sqrt(nsamples)
    assert np.allclose(pmf_gen, pmf_final, atol=atol)


@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("sq_0", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("sq_1", [0.0, 0.1, 1.0])
def test_two_schmidt_mode_guess_exact(eta, sq_0, sq_1):
    """Test that one can invert correctly when there are two Schmidt modes
    and no dark counts.
    """
    pmf = degenerate_pmf({"sq_0": sq_0, "sq_1": sq_1, "n_modes": 2, "eta": eta})
    guess = two_schmidt_mode_guess(pmf)
    assert np.allclose(eta, guess["eta"], atol=1.0e-2)
    sq_ns = [sq_0, sq_1]  # We need to sort sq_n1 and sq_n2 so that sq_n1 >= sq_n2
    sq_0 = np.max(sq_ns)
    sq_1 = np.min(sq_ns)
    assert np.allclose(sq_0, guess["sq_0"], atol=0.1)
    assert np.allclose(sq_1, guess["sq_1"], atol=0.1)


@pytest.mark.parametrize("do_not_vary", ["eta", "noise", None])
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
    if threshold:
        probs = threshold_1d(degenerate_pmf(params), threshold)
    else:
        probs = degenerate_pmf(params)
    fit = fit_1d(probs, params, threshold=threshold, do_not_vary=do_not_vary)
    assert np.allclose(fit.chisqr, 0.0)


def test_marginal_calcs_1d():
    """Tests that marginal_calcs_1d returns the correct values as an array"""
    nmean = 1.0
    ps = degenerate_pmf({"n_modes": 1, "sq_0": nmean})
    assert np.allclose(marginal_calcs_1d(ps, as_dict=False), np.array([nmean, 3 + 1 / nmean]))
