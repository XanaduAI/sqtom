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
"""
Degenerate squeezer inverse-problem solver
==========================================
This module solves the *inverse* problem of given a photon number distribution find the best
of parameters describing the different quantum states in a single beam producing it.

The ideas behind this module borrow heavily on the work of Burenkok et al. in

Full statistical mode reconstruction of a light field via a photon-number-resolved measurement
I. A. Burenkov, A. K. Sharma, T. Gerrits, G. Harder, T. J. Bartley, C. Silberhorn, E. A. Goldschmidt, and S. V. Polyakov
Phys. Rev. A 95, 053806 (2017)
"""


import numpy as np
from lmfit import Minimizer, Parameters
from sqtom.forward_solver import degenerate_pmf


def marginal_calcs_1d(pd_data, as_dict=True):
    """ Given a one dimensional array of probabilities it calculates the mean photon number
    and the g2.

    Args:
        pd_data (array): probability mass function of the photon events
        as_dict (boolean): whether to return the results as a dictionary

    Returns:
        dict or array: values of the mean photons number the corresponding g2.
    """

    intn = pd_data.shape[0]
    n = np.arange(intn)
    nmean = pd_data @ n
    nmean2 = pd_data @ n ** 2
    g2 = (nmean2 - nmean) / nmean ** 2
    if as_dict:
        return {"n": nmean, "g2": g2}
    return np.array([nmean, g2])


def threshold_1d(ps, nmax):
    """ Thresholds a probability distribution by assigning events with nmax
    photons or more to the nmax bin.

    Args:
        ps (array): probability distribution
        nmax (int): threshold value

    Returns:
        array: thresholded probability distribution
    """
    thr = nmax - 1
    local_ps = np.copy(ps)
    local_ps[thr] = np.sum(local_ps[thr:])
    return local_ps[:nmax]


def fit_1d(
    pd_data,
    guess,
    do_not_vary=None,
    method="leastsq",
    threshold=False,
    cutoff=50,
    sq_label="sq_",
    noise_label="noise",
):
    """Takes as input the name of the model to fit to and the jpd of the data
    and returns the fitted model.

    Args:
        pd_data (array): one dimensional array of the probability distribution of the data
        guess (dict): dictionary with the guesses for the different parameters
        method (string): method to be used by the optimizer
        do_not_vary (list): list of variables that should be held constant during optimization
        threshold (boolean or int): whether to threshold the photon probbailitites at value threshold
        cutoff (int): internal cutoff
        sq_label (string): string label for the squeezing parameters
        noise_label (string): label for the noise parameters.

    Returns:
        Object: object containing the optimized parameter and several goodness-of-fit statistics
    """
    if do_not_vary is None:
        do_not_vary = []

    pars_model = Parameters()
    n_modes = guess["n_modes"]
    pars_model.add("n_modes", value=n_modes, vary=False)
    # Add the squeezing parameters
    for i in range(n_modes):
        pars_model.add(sq_label + str(i), value=guess["sq_" + str(i)], min=0.0)

    if "eta" in do_not_vary:
        pars_model.add("eta", value=guess["eta"], vary=False)
    else:
        pars_model.add("eta", value=guess["eta"], min=0.0, max=1.0)

    if noise_label in do_not_vary:
        pars_model.add(noise_label, value=guess[noise_label], vary=False)
    else:
        pars_model.add(noise_label, value=guess[noise_label], min=0.0)

    if threshold:

        def model_1d(params, pd_data):
            ndim = pd_data.shape[0]
            dpmf = degenerate_pmf(params, cutoff=cutoff)
            return threshold_1d(dpmf, ndim) - pd_data

    else:

        def model_1d(params, pd_data):
            ndim = pd_data.shape[0]
            return (
                degenerate_pmf(params, cutoff=cutoff, sq_label=sq_label, noise_label=noise_label)[
                    :ndim
                ]
                - pd_data
            )

    minner_model = Minimizer(model_1d, pars_model, fcn_args=([pd_data]))
    return minner_model.minimize(method=method)
