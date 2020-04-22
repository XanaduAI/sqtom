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
Twin bean forward-problem solver
================================
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
from thewalrus.quantum import loss_mat, gen_single_mode_dist
from scipy.stats import poisson, geom
from scipy.signal import convolve2d

from numba import jit


#pylint: disable=too-many-arguments
def twinbeam_pmf(
    cutoff,
    eta_s=1.0,
    eta_i=1.0,
    poisson_param_ns=0.0,
    poisson_param_ni=0.0,
    twin_bose=None,
):
    r"""  Contructs the joint probability mass function of a conjugate source for a total
    of n photons in both signal idler and for an overall loss after generation
    characterized by the transmissions etas and etai.
    The source is described by either conjugate (correlated) and uncorrelated parts.

    Args:
        cutoff (int): Photon number cutoff
        eta_s (float): Transmission in the signal arm
        eta_i (float): Transmission in idler arm
        poisson_param_ns (float): Mean photon number of Poisson distribution hitting the signal detector
        poisson_param_ni (float): Mean photon number of Poisson distribution hitting the idler detector
        twin_bose (array): Mean photon number(s) of a Bose distribution in the diagonal of the joint probability mass function,
            representing different Schmidt modes

    Returns:
        (array): `n\times n` matrix representing the joint probability mass function

    """

    # First convolve all the 1-d distirbutions.
    ns = poisson.pmf(np.arange(cutoff), poisson_param_ns)
    ni = poisson.pmf(np.arange(cutoff), poisson_param_ni)

    joint_pmf = np.outer(ns, ni)
    # Then convolve with the conjugate distributions if there are any.

    if twin_bose is not None:
        loss_mat_ns = loss_mat(float(eta_s), cutoff).T
        loss_mat_ni = loss_mat(float(eta_i), cutoff)
        twin_pmf = np.zeros([cutoff, cutoff])
        twin_pmf[0, 0] = 1.0
        for nmean in twin_bose:
            twin_pmf = convolve2d(
                twin_pmf, np.diag(geom.pmf(np.arange(1, cutoff + 1), 1 / (1.0 + nmean),)),
            )[0:cutoff, 0:cutoff]

        twin_pmf = loss_mat_ns @ twin_pmf @ loss_mat_ni
        joint_pmf = convolve2d(twin_pmf, joint_pmf)[:cutoff, :cutoff]

    return joint_pmf



def degenerate_pmf(cutoff, sq_n=None, eta=1.0, n_dark=None):
    """Generates the total photon number distribution of single mode squeezed states with different squeezing values.
    After each of them undergoes loss by amount eta
    Args:
        cutoff (int): Fock cutoff
        sq_n (array): array of mean photon numbers of the squeezed modes
        eta (float): Amount of loss
        n_dark (float): mean photon of the mode responsible for dark counts
    Returns:
        (array[int]): total photon number distribution
    """
    ps = np.zeros(cutoff)
    ps[0] = 1.0
    if sq_n is not None:
        mat = loss_mat(float(eta), cutoff)
        for n_val in sq_n:
            ps = np.convolve(ps, gen_single_mode_dist(np.arcsinh(np.sqrt(n_val)), cutoff=cutoff) @ mat)[:cutoff]
    if n_dark is not None:
        ps = np.convolve(ps, poisson.pmf(np.arange(cutoff), n_dark))[:cutoff]
    return ps[:cutoff]
