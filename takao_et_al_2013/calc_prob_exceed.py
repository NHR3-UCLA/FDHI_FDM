"""Functions to calculate the model predictions."""

# Python imports
import sys
import numpy as np

from pathlib import Path
from scipy import stats

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from takao_et_al_2013.functions import (
    _calc_params_ad,
    _calc_params_md,
    _calc_params_d_ad,
    _calc_params_d_md,
)


def func_probex(*, magnitude, location, srl_km, displacement_array, version):
    """
    Calculate the probability of exceedance for a scenario and array of displacement test values.

    Parameters
    ----------
    magnitude : float
        The earthquake moment magnitude.

    location : int or float or numpy.ndarray with a single element
        Normalized location along rupture length, range [0, 1.0].

    srl_km : int or float
        Total surface rupture length in kilometers.

    displacement_array : array-like
        Displacement test values in meters.

    version : str
        TEA13 normalization model name (case-insensitive). Valid options are "d_ad" or "d_md".

    Returns
    -------
    array-like
        The probability of exceedance.
    """

    # User input checks
    valid_versions = {"d_ad", "d_md"}
    if version.lower() not in valid_versions:
        raise ValueError(
            f"Invalid version. Please choose one of the valid options: {valid_versions}."
        )

    for var in [magnitude, location, srl_km]:
        if not isinstance(var, (int, float, np.integer, np.floating)):
            raise TypeError("Only one magnitude, one location, and one SRL is allowed.")

    # Calculate statistical distribution parameter predictions
    if version == "d_ad":
        mu, sigma = _calc_params_ad(magnitude=magnitude)
        alpha, beta = _calc_params_d_ad(location=location, srl_km=srl_km)
        d_xd_cdf_distribution = stats.gamma.cdf
        kwargs = {"a": alpha, "loc": 0, "scale": beta}
    elif version == "d_md":
        mu, sigma = _calc_params_md(magnitude=magnitude)
        alpha, beta = _calc_params_d_md(location=location, srl_km=srl_km)
        d_xd_cdf_distribution = stats.beta.cdf
        kwargs = {"a": alpha, "b": beta}

    ###############################################################################################
    # Use numerical integration to convolve D_XD and XD to get total aleatory variability
    # The probability of exceedance is:
    # $P(D > d) = \int\limits_{all~z} ~[ 1 - F(y)]~f_{AD}(z)~dz$
    # Which can be approximated as:
    # $P(D > d) \approx \sum_{i = z_{min}}^{z_{max}} ~[1 - F(y_i)] ~ f_{AD}(z_i) ~ \Delta z$
    ###############################################################################################

    # Compute array of XD values
    n_eps, dz = 6, 0.1
    epsilons = np.arange(-n_eps, n_eps + dz, dz)
    z = np.power(10, mu + epsilons * sigma)
    prob_z = stats.norm.pdf(epsilons)

    # Compute array of D/XD values
    y = displacement_array / z[:, np.newaxis]
    y = y.T

    # Compute array of D/XD probabilities of exceedance
    ccdf_arr = 1 - d_xd_cdf_distribution(**kwargs, x=y)

    # Compute weighted sum
    ccdf = np.dot(ccdf_arr, prob_z) * dz

    return ccdf
