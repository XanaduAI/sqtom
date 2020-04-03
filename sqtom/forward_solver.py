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
from scipy.stats import poisson, geom, nbinom
from scipy.signal import convolve2d

from numba import jit



@jit(nopython=True)
def loss_mat(eta, cutoff):
    r""" Constructs a binomial loss matrix with transmission eta up to n photons.

    Args:
        eta (float): Transmission coefficient. eta=0.0 means complete loss and eta=1.0 means no loss.
        n (int): photon number cutoff.

    Returns:
        array: :math:`n\times n` matrix representing the loss.

    """
    # If full transmission return the identity
    eta = float(eta)
    if eta < 0.0 or eta > 1.0:
        raise ValueError("The transmission parameter eta should be a number between 0 and 1.")

    if eta == 1.0:
        return np.identity(cutoff)

    # Otherwise construct the matrix elements recursively
    lm = np.zeros((cutoff, cutoff))
    mu = 1.0 - eta
    lm[:, 0] = mu ** (np.arange(cutoff))
    for i in range(cutoff):
        for j in range(1, i + 1):
            lm[i, j] = lm[i, j - 1] * (eta / mu) * (i - j + 1) / (j)
    return lm

#pylint: disable=too-many-arguments
def twinbeam_pmf(
    cutoff,
    eta_s=1.0,
    eta_i=1.0,
    poisson_param_ns=0.0,
    poisson_param_ni=0.0,
    bose_params_ns=None,
    bose_params_ni=None,
    twin_bose=None,
    twin_poisson=None,
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
        bose_params_ns (array): Mean photon number(s) of a Bose distribution hitting the signal detector
        bose_params_ni (array): Mean photon number(s) of a Bose distribution hitting the idler detector
        twin_bose (array): Mean photon number(s) of a Bose distribution in the diagonal of the joint probability mass function,
            representing different Schmidt modes
        twin_poisson (array): Mean photon number(s) of a Poisson distribution in the diagonal of the joint probability mass function.
            *Note:* This should only be used when g^2 is very close to 1.

    Returns:
        (array): `n\times n` matrix representing the joint probability mass function

    """

    # First convolve all the 1-d distirbutions.
    ns = poisson.pmf(np.arange(cutoff), poisson_param_ns)
    ni = poisson.pmf(np.arange(cutoff), poisson_param_ni)

    if bose_params_ns is not None:
        for nmean_ns in bose_params_ns:
            ns = np.convolve(ns, geom.pmf(np.arange(1, cutoff + 1), 1.0 / (1.0 + nmean_ns),),)[:cutoff]
    if bose_params_ni is not None:
        for nmean_ni in bose_params_ni:
            ni = np.convolve(ni, geom.pmf(np.arange(1, cutoff + 1), 1.0 / (1.0 + nmean_ni),),)[:cutoff]

    joint_pmf = np.outer(ns, ni)
    # Then convolve with the conjugate distributions if there are any.
    if twin_bose is not None or twin_poisson is not None:
        loss_mat_ns = loss_mat(float(eta_s), cutoff).T
        loss_mat_ni = loss_mat(float(eta_i), cutoff)
        twin_pmf = np.zeros([cutoff, cutoff])
        twin_pmf[0, 0] = 1.0

        if twin_bose is not None:
            for nmean in twin_bose:
                twin_pmf = convolve2d(
                    twin_pmf, np.diag(geom.pmf(np.arange(1, cutoff + 1), 1 / (1.0 + nmean),)),
                )[0:cutoff, 0:cutoff]

        if twin_poisson is not None:
            for nmean in twin_poisson:
                twin_pmf = convolve2d(twin_pmf, np.diag(poisson.pmf(np.arange(cutoff), nmean)),)[
                    0:cutoff, 0:cutoff
                ]

        twin_pmf = loss_mat_ns @ twin_pmf @ loss_mat_ni
        joint_pmf = convolve2d(twin_pmf, joint_pmf)[:cutoff, :cutoff]

    return joint_pmf


def _gen_single_mode_dist(s, cutoff=50):
    """Generate the photon number distribution of a single mode squeezed state.
    Args:
        s (float): squeezing parameter
        cutoff (int): Fock cutoff
    Returns:
        (array): Photon number distribution
    """
    r = 0.5
    q = 1.0 - np.tanh(s) ** 2
    N = cutoff // 2
    ps_tot = np.zeros(cutoff)
    if cutoff % 2 == 0:
        ps = nbinom.pmf(np.arange(N), p=q, n=r)
        ps_tot[0::2] = ps
    else:
        ps = nbinom.pmf(np.arange(N + 1), p=q, n=r)
        ps_tot[0:-1][0::2] = ps[0:-1]
        ps_tot[-1] = ps[-1]

    return ps_tot


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
            ps = np.convolve(ps, _gen_single_mode_dist(np.arcsinh(np.sqrt(n_val)), cutoff=cutoff) @ mat)[:cutoff]
    if n_dark is not None:
        ps = np.convolve(ps, poisson.pmf(np.arange(cutoff), n_dark))[:cutoff]
    return ps[:cutoff]
