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
    navs = 0.5 * (res["ns"] + res["ni"])
    n1 = 0.5 * (1 + np.sqrt(2 * g2avg - 3)) * Nbar
    n2 = 0.5 * (1 - np.sqrt(2 * g2avg - 3)) * Nbar
    etas = res["ns"] / Nbar
    etai = res["ni"] / Nbar
    noise = np.abs(res["g2s"] - res["g2i"]) * Nbar
    return {"etas": etas, "etai": etai, "twin_n1": n1, "twin_n2": n2, "ns": noise, "ni": noise}



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

