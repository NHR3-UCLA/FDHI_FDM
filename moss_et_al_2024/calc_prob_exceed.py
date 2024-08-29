"""Functions to calculate the model predictions."""

# Python imports
import sys
import numpy as np

from pathlib import Path
from scipy import stats

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moss_et_al_2024.functions import (
    _calc_params_ad,
    _calc_params_md,
    _calc_params_d_ad,
    _calc_params_d_md,
)


def func_probex(*, magnitude, location, displacement_array, version):
    """
    Calculate the probability of exceedance for a scenario and array of displacement test values.

    Parameters
    ----------
    magnitude : float
        The earthquake moment magnitude.

    location : int or float or numpy.ndarray with a single element
        Normalized location along rupture length, range [0, 1.0].

    displacement_array : array-like
        Displacement test values in meters.

    version : str
        MEA22 normalization model name (case-insensitive). Valid options are "d_ad" or "d_md".

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

    for var in [magnitude, location]:
        if not isinstance(var, (int, float, np.integer, np.floating)):
            raise TypeError("Only one magnitude and one location is allowed.")

    # Calculate statistical distribution parameter predictions
    if version == "d_ad":
        mu, sigma = _calc_params_ad(magnitude=magnitude)
        alpha, beta = _calc_params_d_ad(location=location)
    elif version == "d_md":
        mu, sigma = _calc_params_md(magnitude=magnitude)
        alpha, beta = _calc_params_d_md(location=location)

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
    cdf = stats.gamma.cdf(x=y, a=alpha, loc=0, scale=beta)

    # Truncation to correct for D/MD >1
    if version == "d_md":
        cdf_1 = stats.gamma.cdf(x=1, a=alpha, loc=0, scale=beta)
        ccdf_arr = np.where(y > 1, 0, (1 - cdf / cdf_1))
    else:
        ccdf_arr = 1 - cdf

    # Compute weighted sum
    ccdf = np.dot(ccdf_arr, prob_z) * dz

    return ccdf
