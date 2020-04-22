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
Twin-beam squeezer inverse-problem solver
=========================================
This module solves the *inverse* problem of given a joint photon number distribution find the best
parameters describing different the quantum states in a twin-beam producing it.

The ideas behind this module borrow are re-implementation of the ideas in

Full statistical mode reconstruction of a light field via a photon-number-resolved measurement
I. A. Burenkov, A. K. Sharma, T. Gerrits, G. Harder, T. J. Bartley, C. Silberhorn, E. A. Goldschmidt, and S. V. Polyakov
Phys. Rev. A 95, 053806 (2017)
"""

import numpy as np
from lmfit import Minimizer, Parameters
from sqtom.forward_solver import twinbeam_pmf

def two_schmidt_mode_guess(jpd_data):
    """Given a two mode histogram, this function generates a "physically" motivated guess for the loss, Schmidt occupations
    and dark counts parameters.
    This model is sensible only if the average g2 of signal and idler is above 1.5.

    Args:
        jpd_data (array): rectangular array with the probability mass functions of the photon events

    Returns:
        dict: Dictionary containing a set of "reasonable" model parameters.
    """
    res = marginal_calcs_2d(jpd_data)
    g2avg = np.max([0.5 * (res["g2_s"] + res["g2_i"]), 1.5])
    Nbar = 1.0 / (res["g11"] - g2avg)
    n0 = 0.5 * (1 + np.sqrt(2 * g2avg - 3)) * Nbar
    n1 = 0.5 * (1 - np.sqrt(2 * g2avg - 3)) * Nbar
    etas = res["n_s"] / Nbar
    etai = res["n_i"] / Nbar
    noise = np.abs(res["g2_s"] - res["g2_i"]) * Nbar
    return {"eta_s": etas, "eta_i": etai, "sq_0": n0, "sq_1": n1, "noise_s": noise * etas, "noise_i": noise * etai}



def marginal_calcs_2d(jpd_data, as_dict=True):
    """ Given a two dimensional array of probabilities it calculates the first
    two moments of the marginal distributions and also their g2's and g11
    It returns these values as a dictionary

    Args:
        jpd_data (array): rectangular array with the probability mass functions of the photon events

    Returns:
        dict or list: values of the mean photons number for signal and idlers, their corresponding g2 and their g11.
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
        return {"n_s": ns, "n_i": ni, "g11": g11, "g2_s": g2s, "g2_i": g2i, "n_modes":2}
    return np.array([n_s, n_i, g11, g2_s, g2_i])

def gen_hist_2d(beam1, beam2):
    """Calculate the joint probability mass function of joint events.
    Args:
        beam1 (array): 1D events array containing the raw click events of first beam
        beam2 (array): 1D events array containing the raw click events of second beam
    Returns:
        (array): probability mass function of the click patterns in vals.
    """
    nx = np.max(beam1)
    ny = np.max(beam2)
    xedges = np.arange(nx + 2)
    yedges = np.arange(ny + 2)
    mass_fun, _, _ = np.histogram2d(beam1, beam2, bins=(xedges, yedges), normed=True)
    return mass_fun



def model_2d(params, pd_data, n_max=50):
    """Constructs a joint probability distribution (jpd) given squeezer parameters
    like noise, squeezing values and noise and the returns the difference between
    the constructed jpd and the the jpd of the data that is to be fit.
    Args:
        params (dict): dictionary of all the Parameter objects required to specify a fit model
        jpd_data (array): rectangular array with the probabilities of the photon events
    Returns:
        (array): rectangular array with the difference between the calculated model and pd_data
    """
    (dim_s,dim_i) = pd_data.shape

    n_modes = int(params["n_modes"])
    sq_n = [params["sq_n" + str(i)] for i in range(n_modes)]
    etai = params["etai"]
    etas = params["etas"]
    ns = params["ns"]
    ni = params["ni"]

    if n_max in params:
        n_max = params["n_max"]
    else:
        n_max=40
    model_pmf = twinbeam_pmf(n_max, eta_s=etas, eta_i=etai, twin_bose=sq_n, poisson_param_ns=ns, poisson_param_ni=ni)[0:dim_s, 0:dim_i]
    #if "threshold" in params:
    #    threshold = int(params["threshold"])-1
    #    model_pmf[threshold] = np.sum(model_pmf[threshold:])
    #    model_pmf[(1+threshold):] = 0.0

    return model_pmf - pd_data



def fit_2d(pd_data, guess, do_not_vary=[], method="leastsq", threshold=False, cutoff=50):
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
    for i in range(n_modes):
        pars_model.add("sq_" + str(i), value=guess["sq_" + str(i)], min=0.0)


    if "eta_s" in do_not_vary:
        pars_model.add("eta_s", value=guess["eta_s"], vary=False)
    else:
        pars_model.add("eta_s", value=guess["eta_s"], min=0.0, max=1.0)

    if "eta_i" in do_not_vary:
        pars_model.add("eta_i", value=guess["eta_i"], vary=False)
    else:
        pars_model.add("eta_i", value=guess["eta_i"], min=0.0, max=1.0)


    if "noise_s" in do_not_vary:
        pars_model.add("noise_s", value=guess["noise_s"], vary=False)
    else:
        pars_model.add("noise_s", value=guess["noise_s"], min=0.0)

    if "noise_i" in do_not_vary:
        pars_model.add("noise_i", value=guess["noise_i"], vary=False)
    else:
        pars_model.add("noise_i", value=guess["noise_i"], min=0.0)

    #if "threshold" in guess:
    #    pars_model.add("threshold", value=guess["threshold"], vary=False)
    # Add the squeezing parameters
    def model_2d(params, jpd_data):
        (dim_s, dim_i) = pd_data.shape
        return twinbeam_pmf(params, cutoff=cutoff)[:dim_s, :dim_i] - pd_data


    minner_model = Minimizer(model_2d, pars_model, fcn_args=([pd_data]))
    result_model = minner_model.minimize(method=method)

    return result_model
