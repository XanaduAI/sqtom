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
Twin-beam squeezer inverse-problem solver
=========================================
This module solves the *inverse* problem: given a joint photon number distribution find the best
parameters describing the different quantum states in a twin-beam producing it.

The ideas behind this module borrow heavily on the work of Burenkok et al. in

Full statistical mode reconstruction of a light field via a photon-number-resolved measurement
I. A. Burenkov, A. K. Sharma, T. Gerrits, G. Harder, T. J. Bartley, C. Silberhorn, E. A. Goldschmidt, and S. V. Polyakov
Phys. Rev. A 95, 053806 (2017)
"""

import numpy as np
from lmfit import Minimizer, Parameters
from sqtom.forward_solver import twinbeam_pmf


def two_schmidt_mode_guess(jpd_data, sq_label="sq_", noise_label="noise"):
    """Given a two mode histogram, this function generates a "physically" motivated guess for the loss, Schmidt occupations
    and dark counts parameters.

    This model is sensible only if the average g2 of signal and idler is above 1.5.

    Args:
        jpd_data (array): rectangular array with the probability mass functions of the photon events
        sq_label (string): label for the squeezing parameters.
        noise_label (string): label for the noise parameters.

    Returns:
        dict: dictionary containing a set of "reasonable" model parameters
    """
    res = marginal_calcs_2d(jpd_data)
    g2avg = np.max([0.5 * (res["g2_s"] + res["g2_i"]), 1.5])
    Nbar = 1.0 / (res["g11"] - g2avg)
    n0 = 0.5 * (1 + np.sqrt(2 * g2avg - 3)) * Nbar
    n1 = 0.5 * (1 - np.sqrt(2 * g2avg - 3)) * Nbar
    etas = res["n_s"] / Nbar
    etai = res["n_i"] / Nbar
    noise = np.abs(res["g2_s"] - res["g2_i"]) * Nbar
    return {
        "eta_s": etas,
        "eta_i": etai,
        sq_label + "0": n0,
        sq_label + "1": n1,
        noise_label + "_s": noise * etas,
        noise_label + "_i": noise * etai,
        "n_modes": 2,
    }


def marginal_calcs_2d(jpd_data, as_dict=True):
    """Given a two dimensional array of probabilities it calculates
    the mean photon numbers, their g2's and g11.

    It returns these values as a dictionary or as an array.

    Args:
        jpd_data (array): probability mass function of the photon events
        as_dict (boolean): whether to return the results as a dictionary

    Returns:
        dict or array: values of the mean photons number for signal and idlers, their corresponding g2 and their g11
    """
    inta, intb = jpd_data.shape
    na = np.arange(inta)
    nb = np.arange(intb)
    ns = np.sum(jpd_data, axis=1) @ na
    ni = np.sum(jpd_data, axis=0) @ nb
    ns2 = np.sum(jpd_data, axis=1) @ (na ** 2)
    ni2 = np.sum(jpd_data, axis=0) @ (nb ** 2)
    g2s = (ns2 - ns) / ns ** 2
    g2i = (ni2 - ni) / ni ** 2
    g11 = (na @ jpd_data @ nb) / (ns * ni)
    if as_dict is True:
        return {
            "n_s": ns,
            "n_i": ni,
            "g11": g11,
            "g2_s": g2s,
            "g2_i": g2i,
        }
    return np.array([ns, ni, g11, g2s, g2i])


def gen_hist_2d(beam1, beam2):
    """Calculate the joint probability mass function of events.

    Args:
        beam1 (array): 1D events array containing the raw click events of first beam
        beam2 (array): 1D events array containing the raw click events of second beam

    Returns:
        array: probability mass function of the click patterns for the two beams
    """
    nx = np.max(beam1)
    ny = np.max(beam2)
    xedges = np.arange(nx + 2)
    yedges = np.arange(ny + 2)
    mass_fun, _, _ = np.histogram2d(beam1, beam2, bins=(xedges, yedges), normed=True)
    return mass_fun


def fit_2d(
    pd_data, guess, do_not_vary=[], method="leastsq", cutoff=50, sq_label="sq_", noise_label="noise"
):
    """Returns a model fit from the parameter guess and the data

    Args:
        pd_data (array): one dimensional array of the probability distribution of the data
        guess (dict): dictionary with the guesses for the different parameters
        method (string): method to be used by the optimizer
        do_not_vary (list): list of variables that should be held constant during optimization
        cutoff (int): internal cutoff
        sq_label (string): label for the squeezing parameters.
        noise_label (string): label for the noise parameters.

    Returns:
        Object: object containing the optimized parameter and several goodness-of-fit statistics
    """
    pars_model = Parameters()
    n_modes = guess["n_modes"]
    pars_model.add("n_modes", value=n_modes, vary=False)
    for i in range(n_modes):
        pars_model.add(sq_label + str(i), value=guess["sq_" + str(i)], min=0.0)

    if "eta_s" in do_not_vary:
        pars_model.add("eta_s", value=guess["eta_s"], vary=False)
    else:
        pars_model.add("eta_s", value=guess["eta_s"], min=0.0, max=1.0)

    if "eta_i" in do_not_vary:
        pars_model.add("eta_i", value=guess["eta_i"], vary=False)
    else:
        pars_model.add("eta_i", value=guess["eta_i"], min=0.0, max=1.0)

    if noise_label + "_s" in do_not_vary:
        pars_model.add(noise_label + "_s", value=guess[noise_label + "_s"], vary=False)
    else:
        pars_model.add(noise_label + "_s", value=guess[noise_label + "_s"], min=0.0)

    if noise_label + "_i" in do_not_vary:
        pars_model.add(noise_label + "_i", value=guess[noise_label + "_i"], vary=False)
    else:
        pars_model.add(noise_label + "_i", value=guess[noise_label + "_i"], min=0.0)

    def model_2d(params, jpd_data):
        (dim_s, dim_i) = pd_data.shape
        joint_pmf = twinbeam_pmf(params, cutoff=cutoff)[:dim_s, :dim_i]
        return joint_pmf - pd_data

    minner_model = Minimizer(model_2d, pars_model, fcn_args=([pd_data]))
    result_model = minner_model.minimize(method=method)

    return result_model
