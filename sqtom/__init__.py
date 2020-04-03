from ._fitting_1d import marginal_calcs_1d, model_1d, fit_1d
from .fitting_2d import marginal_calcs_2d, two_schmidt_mode_guess, gen_hist_2d
from ._photon_tom import loss_mat, degenerate_pmf, twinbeam_pmf


def version():
    r"""
    Get version number of sqtom
    Returns:
      str: The package version number
    """
    return __version__