import numpy as np
from lmfit import Minimizer, Parameters
from ._photon_tom import degenerate_pmf


def marginal_calcs_1d(pd_data, as_dict=True):
    """ Given a two dimensional array of probabilities it calculates the first
    two moments of the marginal distributions and also their g2's and g11
    It returns these values as a dictionary"""

    (intn,) = pd_data.shape
    n = np.arange(intn)
    nmean = pd_data @ n
    nmean2 = pd_data @ n ** 2
    g2 = (nmean2 - nmean) / nmean ** 2
    if as_dict is True:
        return {"n": nmean, "g2": g2}
    return np.array([nmean, g2])


def model_1d(params, pd_data, n_max=50):
    """Constructs a joint probability distribution (jpd) given squeezer parameters
    like noise, squeezing values and noise and the returns the difference between
    the constructed jpd and the the jpd of the data that is to be fit.
    Args:
        params (dict): dictionary of all the Parameter objects required to specify a fit model
        pd_data (array): rectangular array with the probability mass functions of the photon events
        model_name (str): describes the model used for fitting
    Returns:
        (array): rectangular array with the difference between the calculated model and pd_data
    """
    (dim,) = pd_data.shape
    n_modes = int(params["n_modes"])
    sq_n = [params["sq_n" + str(i)] for i in range(n_modes)]
    eta = params["eta"]
    n_dark = params["n_dark"]
    model_pmf = degenerate_pmf(n_max, eta=eta, sq_n=sq_n, n_dark=n_dark)[0:dim]

    return model_pmf - pd_data


def fit_1d(model_name, pd_data, guess, method="leastsq"):
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
    # Add the squeezing parameters
    for i in range(n_modes):
        pars_model.add("sq_n" + str(i), value=guess["sq_n" + str(i)], min=0.0)

    if model_name == "NoiseFixed":
        pars_model.add("eta", value=guess["eta"], min=0.0, max=1.0)
        pars_model.add("n_dark", value=guess["n_dark"], vary=False)
    elif model_name == "LossFixed":
        pars_model.add("eta", value=guess["eta"], vary=False)
        pars_model.add("n_dark", value=guess["n_dark"], min=0.0)
    elif model_name == "NoiseLossFixed":
        pars_model.add("eta", value=guess["eta"], vary=False)
        pars_model.add("n_dark", value=guess["n_dark"], vary=False)
    elif model_name == "NoneFixed":
        pars_model.add("eta", value=guess["eta"], min=0.0, max=1.0)
        pars_model.add("n_dark", value=guess["n_dark"], min=0.0)
    else:
        raise NotImplementedError("Model " + model_name + " not implemented.")

    minner_model = Minimizer(model_1d, pars_model, fcn_args=([pd_data]))
    result_model = minner_model.minimize(method=method)

    return result_model
