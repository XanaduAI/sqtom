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
from scipy.optimize import root_scalar
from lmfit import Minimizer, Parameters
from sqtom.forward_solver import degenerate_pmf


def two_schmidt_mode_guess(pd_data, sq_label="sq_", noise_fraction=0.001):
    """Given a single mode histogram, this function generates a "physically" motivated guess for the loss, Schmidt occupations
    and dark counts parameters.

    This model is sensible only if the average g2 is above 2.

    Args:
        pd_data (array): rectangular array with the probability mass functions of the photon events
        sq_label (string): label for the squeezing parameters.

    Returns:
        dict: dictionary containing a set of "reasonable" model parameters
    """
    res = marginal_calcs_1d(pd_data)
    nmean = res["n"]
    g2 = res["g2"]
    P0 = pd_data[0]

    def findeta(eta, nmean, g2, P0):
        a = nmean / 4
        b = (3 - g2) * (nmean ** 2) / 4 - nmean
        c = (g2 - 3) * nmean ** 2
        d = (3 - g2) * nmean ** 2 + 2 * nmean + 1 - 1 / P0 ** 2
        return a * eta ** 3 + b * eta ** 2 + c * eta + d

    eta_set = np.linspace(-0.01, 1.01, num=52)
    function_root_search = np.array([findeta(i, nmean, g2, P0) for i in eta_set])
    indices = np.array([])
    for i in range(function_root_search.size - 1):
        if function_root_search[i + 1] / function_root_search[i] < 0:
            indices = np.append(indices, i)
            indices = np.append(indices, i + 1)
    eta = root_scalar(
        findeta,
        args=(nmean, g2, P0),
        bracket=(eta_set[int(indices[-1] - 1)], eta_set[int(indices[-1])]),
    ).root
    if eta > 1:
        eta = 1
    if eta < 0:
        eta = 0
    if (g2 - 2) * nmean ** 2 - eta * nmean >= 0 and eta > 0:
        n0 = (nmean + np.sqrt((g2 - 2) * nmean ** 2 - eta * nmean)) / (2 * eta)
        n1 = (nmean - np.sqrt((g2 - 2) * nmean ** 2 - eta * nmean)) / (2 * eta)
    elif (g2 - 2) * nmean ** 2 - eta * nmean < 0 and eta > 0:
        n0 = nmean / (2 * eta)
        n1 = nmean / (2 * eta)
    else:
        n0 = 0
        n1 = 0
    noise = nmean * noise_fraction
    return {
        "eta": eta,
        sq_label + "0": n0,
        sq_label + "1": n1,
        "noise": noise,
        "n_modes": 2,
    }


def marginal_calcs_1d(pd_data, as_dict=True):
    """Given a one dimensional array of probabilities it calculates the mean photon number
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


def gen_hist_1d(beam):
    """Calculate the probability mass function of events.

    Args:
        beam (array): 1D events array containing the raw click events of the beam

    Returns:
        array: probability mass function of the click patterns for the beam
    """
    nmax = int(np.max(beam))
    return np.histogram(beam, bins=nmax, density=True)[0]


def threshold_1d(ps, nmax):
    """Thresholds a probability distribution by assigning events with nmax
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
