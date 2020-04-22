# Copyright 2019 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Basic tests for the functions in fitting_2d"""

import pytest
import numpy as np
from sqtom.forward_solver import twinbeam_pmf
from sqtom.fitting_2d import fit_2d, gen_hist_2d, two_schmidt_mode_guess



@pytest.mark.parametrize("sq_0", [0.1, 1.0, 2.0])
def test_gen_hist_2d_twin(sq_0):
    """Check that a histogram is constructed correctly for a lossless pure twin-beam source"""
    nsamples = 1000000
    p = 1 / (1.0 + sq_0)
    samples = np.random.geometric(p, nsamples) - 1
    mat = gen_hist_2d(samples, samples)
    n, m = mat.shape
    assert n == m
    expected = twinbeam_pmf({"sq_0":sq_0, "n_modes":1}, cutoff=n)
    assert np.allclose(mat, expected, atol=0.01)

@pytest.mark.parametrize("ns", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("ni", [0.1, 1.0, 2.0])
def test_gen_hist_2d_poisson(ns, ni):
    """Test the histograms are correctly generated for pure noise"""
    nsamples = 1000000
    mat = gen_hist_2d(np.random.poisson(ns, nsamples), np.random.poisson(ni, nsamples))
    n, m = mat.shape
    nmax = np.max([n, m])
    expected = twinbeam_pmf({"noise_s":ns, "noise_i":ni})[:n, :m]
    np.allclose(expected, mat, atol=0.01)
@pytest.mark.parametrize("eta_s", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("eta_i", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("sq_0", [0.0, 0.1, 1.0, 2.0])
@pytest.mark.parametrize("sq_1", [0.1, 1.0, 2.0])
def test_two_schmidt_mode_guess_exact(eta_s, eta_i, sq_0, sq_1):
    """Test that one can invert correctly when there are two Schmidt modes
    and no dark counts.
    """
    pmf = twinbeam_pmf({"sq_0":sq_0, "sq_1":sq_1, "n_modes":2, "eta_s":eta_s, "eta_i":eta_i})
    guess = two_schmidt_mode_guess(pmf)
    assert np.allclose(eta_s, guess["eta_s"], atol=1.0e-2)
    assert np.allclose(eta_i, guess["eta_i"], atol=1.0e-2)
    sq_ns = [sq_0, sq_1]  # We need to sort sq_n1 and sq_n2 so that sq_n1 >= sq_n2
    sq_0 = np.max(sq_ns)
    sq_1 = np.min(sq_ns)
    assert np.allclose(sq_0, guess["sq_0"], atol=1.0e-2)
    assert np.allclose(sq_1, guess["sq_1"], atol=1.0e-2)
