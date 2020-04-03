import pytest
import numpy as np
from sqtom import degenerate_pmf, model_1d, fit_1d


@pytest.mark.parametrize("model", ["NoneFixed", "LossFixed", "NoiseFixed", "NoiseLossFixed"])
@pytest.mark.parametrize("n_modes", [1,2,3])
def test_exact_model(model, n_modes):
	"""Test that the fitting is correct when the guess is exactly the correct answer"""
	sq_n = 0.7 * (0.5 ** np.arange(n_modes))
	n_dark = 0.1
	eta = 0.7
	probs = degenerate_pmf(50, sq_n=sq_n, eta=eta, n_dark=n_dark)
	guess = {"sq_n" + str(i):sq_n[i] for i in range(n_modes)}
	guess["eta"] = eta
	guess["n_dark"] = n_dark
	guess["n_modes"] = n_modes
	fit = fit_1d(model, probs, guess)
	assert np.allclose(fit.chisqr, 0.0)
