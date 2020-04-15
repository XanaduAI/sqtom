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
"""
Degenerate squeezer inverse-problem solver
==========================================
This module solves the *inverse* problem of given a photon number distribution find the best
of parameters describing different the quantum states in a single beam producing it.

The ideas behind this module borrow heavily on the work of Burenkok et al. in

Full statistical mode reconstruction of a light field via a photon-number-resolved measurement
I. A. Burenkov, A. K. Sharma, T. Gerrits, G. Harder, T. J. Bartley, C. Silberhorn, E. A. Goldschmidt, and S. V. Polyakov
Phys. Rev. A 95, 053806 (2017)
"""


import numpy as np
from lmfit import Minimizer, Parameters
from sqtom.forward_solver import degenerate_pmf


def marginal_calcs_1d(pd_data, as_dict=True):
    """ Given a two dimensional array of probabilities it calculates the first
    two moments of the marginal distributions and also their g2's and g11
    It returns these values as a dictionary"""

    (intn,) = pd_data.shape
    n = np.arange(intn)
    nmean = pd_data @ n
    nmean2 = pd_data @ n ** 2
    g2 = (nmean2 - nmean) / nmean ** 2
    if as_dict is True:
        return {"n": nmean, "g2": g2}
    return np.array([nmean, g2])


def model_1d(params, pd_data, n_max=50):
    """Constructs a joint probability distribution (jpd) given squeezer parameters
    like noise, squeezing values and noise and the returns the difference between
    the constructed jpd and the the jpd of the data that is to be fit.
    Args:
        params (dict): dictionary of all the Parameter objects required to specify a fit model
        pd_data (array): rectangular array with the probability mass functions of the photon events
        model_name (str): describes the model used for fitting
    Returns:
        (array): rectangular array with the difference between the calculated model and pd_data
    """
    (dim,) = pd_data.shape

    n_modes = int(params["n_modes"])
    sq_n = [params["sq_n" + str(i)] for i in range(n_modes)]
    eta = params["eta"]
    n_dark = params["n_dark"]
    model_pmf = degenerate_pmf(n_max, eta=eta, sq_n=sq_n, n_dark=n_dark)[0:dim]
    if "threshold" in params:
    	threshold = int(params["threshold"])-1
    	model_pmf[threshold] = np.sum(model_pmf[threshold:])
    	model_pmf[(1+threshold):] = 0.0

    return model_pmf - pd_data


def fit_1d(pd_data, guess, method="leastsq", do_not_vary=[]):
    """Takes as input the name of the model to fit to and the jpd of the data
    and returns the fitted model.
    Args:
        model_name (str): describes the model used for fitting
        pd_data (array): one dimensional array of the probability distribution of the data
        guess (dict): dictionary with the guesses for the different parameters
    Returns:
        Object containing the optimized parameter and several goodness-of-fit statistics
    """
    pars_model = Parameters()
    n_modes = guess["n_modes"]
    pars_model.add("n_modes", value=n_modes, vary=False)
    if "threshold" in guess:
    	pars_model.add("threshold", value=guess["threshold"], vary=False)
    # Add the squeezing parameters
    for i in range(n_modes):
        pars_model.add("sq_n" + str(i), value=guess["sq_n" + str(i)], min=0.0)
    params = ["eta", "n_dark"]


    if "eta" in do_not_vary:
    	pars_model.add("eta", value=guess["eta"], vary=False)
    else:
    	pars_model.add("eta", value=guess["eta"], min=0.0, max=1.0)

    if "n_dark" in do_not_vary:
    	pars_model.add("n_dark", value=guess["n_dark"], vary=False)
    else:
    	pars_model.add("n_dark", value=guess["n_dark"], min=0.0)

    minner_model = Minimizer(model_1d, pars_model, fcn_args=([pd_data]))
    result_model = minner_model.minimize(method=method)

    return result_model
