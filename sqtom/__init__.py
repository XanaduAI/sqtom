from ._fitting_1d import marginal_calcs_1d, model_1d, fit_1d
from ._photon_tom import loss_mat, twin_beam_pmf, degenerate_pmf


def version():
    r"""
    Get version number of sqtom
    Returns:
      str: The package version number
    """
    return __version__