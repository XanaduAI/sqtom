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
Twin-beam and degenerate forward-problem solver
===============================================
This module solves the *forward* problem of given a set of parameters describing different
quantum state in two beams producing the joint photon number distribution. This joint
photon number distribution can then be passed to lmfit to solve the inverse problem using
the Levenberg-Marquardt method.

The ideas behind this module borrow heavily on the work of Burenkok et al. in

Full statistical mode reconstruction of a light field via a photon-number-resolved measurement
I. A. Burenkov, A. K. Sharma, T. Gerrits, G. Harder, T. J. Bartley, C. Silberhorn, E. A. Goldschmidt, and S. V. Polyakov
Phys. Rev. A 95, 053806 (2017)
"""


import numpy as np
from scipy.stats import poisson, geom
from scipy.signal import convolve2d
from thewalrus.quantum import loss_mat, gen_single_mode_dist


def twinbeam_pmf(params, cutoff=50, sq_label="sq_", noise_label="noise"):
    r"""Contructs the joint probability mass function of a conjugate source.

    Args:
        params (dict): Parameter dictionary, with possible keys "noise_s", "noise_i" for the
        Poisson noise mean photon numbers, "eta_s", "eta_i" for the transmission of the twin_beams,
        "n_modes" describing the number of twin_beams and the parameters sq_0,..,sq_n where
        n = n_modes giving the mean photon numbers of the different twin_beams.
        cutoff (int): Fock cutoff.
        sq_label (string): label for the squeezing parameters.
        noise_label (string): label for the noise parameters.

    Returns:
        (array): `n\times n` matrix representing the joint probability mass function
    """
    if noise_label + "_s" in params:
        noise_s = float(params[noise_label + "_s"])
    else:
        noise_s = 0.0

    if noise_label + "_i" in params:
        noise_i = float(params[noise_label + "_i"])
    else:
        noise_i = 0.0

    if "eta_s" in params:
        eta_s = float(params["eta_s"])
    else:
        eta_s = 1.0

    if "eta_i" in params:
        eta_i = float(params["eta_i"])
    else:
        eta_i = 1.0

    # First convolve all the 1-d distributions.
    ns = poisson.pmf(np.arange(cutoff), noise_s)
    ni = poisson.pmf(np.arange(cutoff), noise_i)

    joint_pmf = np.outer(ns, ni)
    # Then convolve with the twin beam distributions if there are any.
    if "n_modes" in params:
        n_modes = int(params["n_modes"])
        sq = [float(params[sq_label + str(i)]) for i in range(n_modes)]
        loss_mat_ns = loss_mat(eta_s, cutoff).T
        loss_mat_ni = loss_mat(eta_i, cutoff)
        twin_pmf = np.zeros([cutoff, cutoff])
        twin_pmf[0, 0] = 1.0
        for nmean in sq:
            twin_pmf = convolve2d(
                twin_pmf, np.diag(geom.pmf(np.arange(1, cutoff + 1), 1 / (1.0 + nmean),)),
            )[0:cutoff, 0:cutoff]
        twin_pmf = loss_mat_ns @ twin_pmf @ loss_mat_ni
        joint_pmf = convolve2d(twin_pmf, joint_pmf)[:cutoff, :cutoff]

    return joint_pmf


def degenerate_pmf(params, cutoff=50, sq_label="sq_", noise_label="noise"):
    r"""Contructs the probability mass function of a degenerate squeezing source.

    Args:
        params (dict): Parameter dictionary, with possible keys "noise" for the
        Poisson noise mean photon number, "eta", for the loss transmission, "n_modes" 
        describing the number of squeezed modes and the parameters sq_0,..,sq_n where
        n = n_modes giving the mean photon numbers of the different squeezers.
        cutoff (int): Fock cutoff.
        sq_label (string): label for the squeezing parameters.
        noise_label (string): label for the noise parameters.

    Returns:
        (array): `n\times n` matrix representing the joint probability mass function
    """
    if noise_label in params:
        noise = float(params[noise_label])
    else:
        noise = 0.0

    if "eta" in params:
        eta = float(params["eta"])
    else:
        eta = 1.0

    ps = poisson.pmf(np.arange(cutoff), noise)

    if "n_modes" in params:
        n_modes = int(params["n_modes"])
        sq = [float(params[sq_label + str(i)]) for i in range(n_modes)]
        mat = loss_mat(float(eta), cutoff)
        for n_val in sq:
            ps = np.convolve(
                ps, gen_single_mode_dist(np.arcsinh(np.sqrt(n_val)), cutoff=cutoff) @ mat,
            )[:cutoff]

    return ps[:cutoff]
