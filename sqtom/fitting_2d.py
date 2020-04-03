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
    g2avg = np.max([0.5 * (res["g2s"] + res["g2i"]), 1.5])
    Nbar = 1.0 / (res["g11"] - g2avg)
    n1 = 0.5 * (1 + np.sqrt(2 * g2avg - 3)) * Nbar
    n2 = 0.5 * (1 - np.sqrt(2 * g2avg - 3)) * Nbar
    etas = res["ns"] / Nbar
    etai = res["ni"] / Nbar
    noise = np.abs(res["g2s"] - res["g2i"]) * Nbar
    return {"etas": etas, "etai": etai, "twin_n1": n1, "twin_n2": n2, "ns": noise * etas, "ni": noise * etai}



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
        return {"ns": ns, "ni": ni, "g11": g11, "g2s": g2s, "g2i": g2i}
    return np.array([ns, ni, g11, g2s, g2i])

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
    mass_fun, xedges, yedges = np.histogram2d(beam1, beam2, bins=(xedges, yedges), normed=True)
    return mass_fun
