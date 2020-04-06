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
"""Basic tests for the functions in fitting_1d"""

import pytest
import numpy as np
from sqtom.forward_solver import degenerate_pmf
from sqtom.fitting_1d import fit_1d

@pytest.mark.parametrize("model_name", ["NoneFixed", "LossFixed", "NoiseFixed", "NoiseLossFixed"])
@pytest.mark.parametrize("n_modes", [1, 2, 3])
def test_exact_model_1d(model_name, n_modes):
    """Test that the fitting is correct when the guess is exactly the correct answer"""
    sq_n = 0.7 * (0.5 ** np.arange(n_modes))
    n_dark = 0.1
    eta = 0.7
    probs = degenerate_pmf(50, sq_n=sq_n, eta=eta, n_dark=n_dark)
    guess = {"sq_n" + str(i):sq_n[i] for i in range(n_modes)}
    guess["eta"] = eta
    guess["n_dark"] = n_dark
    guess["n_modes"] = n_modes
    fit = fit_1d(model_name, probs, guess)
    assert np.allclose(fit.chisqr, 0.0)


def test_not_known_mode():
    n_modes = 1
    model_name = "SphericalCow"
    sq_n = 0.7 * (0.5 ** np.arange(n_modes))
    n_dark = 0.1
    eta = 0.7
    probs = degenerate_pmf(50, sq_n=sq_n, eta=eta, n_dark=n_dark)
    guess = {"sq_n" + str(i):sq_n[i] for i in range(n_modes)}
    guess["eta"] = eta
    guess["n_dark"] = n_dark
    guess["n_modes"] = n_modes
    with pytest.raises(NotImplementedError, match="Model " + model_name + " not implemented."):
        fit_1d(model_name, probs, guess)
