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


def func_det(*, magnitude, location, percentile, srl_km, version):
    """
    Calculate the displacement in meters for a scenario.

    Parameters
    ----------
    magnitude : int or float or numpy.ndarray with a single element
        Earthquake moment magnitude.

    location : int or float or numpy.ndarray with a single element
        Normalized location along rupture length, range [0, 1.0].

    percentile : int or float or numpy.ndarray with a single element
        Aleatory quantile value. Use -1 for mean.

    srl_km : int or float
        Total surface rupture length in kilometers.

    version : str
        TEA13 normalization model name (case-insensitive). Valid options are "d_ad" or "d_md".

    Returns
    -------
    float
        The displacement in meters.
    """

    # User input checks
    valid_versions = {"d_ad", "d_md"}
    if version.lower() not in valid_versions:
        raise ValueError(
            f"Invalid version. Please choose one of the valid options: {valid_versions}."
        )

    for var in [magnitude, location, srl_km, percentile]:
        if not isinstance(var, (int, float, np.integer, np.floating)):
            raise TypeError(
                "Only one magnitude, one location, one SRL, and one percentile is allowed."
            )

    # Calculate statistical distribution parameter predictions
    if version == "d_ad":
        mu, sigma = _calc_params_ad(magnitude=magnitude)
        alpha, beta = _calc_params_d_ad(location=location, srl_km=srl_km)
        d_xd_distribution = stats.gamma.rvs
        kwargs = {"a": alpha, "loc": 0, "scale": beta}
    elif version == "d_md":
        mu, sigma = _calc_params_md(magnitude=magnitude)
        alpha, beta = _calc_params_d_md(location=location, srl_km=srl_km)
        d_xd_distribution = stats.beta.rvs
        kwargs = {"a": alpha, "b": beta}

    # Use sampling to convolve D_XD and XD to get total aleatory variability
    N = 500_000
    np.random.seed(1)
    samples_xd = np.power(10, stats.norm.rvs(loc=mu, scale=sigma, size=N))
    np.random.seed(1)  # this needs to be reset before each RVS
    samples_d_xd = d_xd_distribution(**kwargs, size=N)

    # Combine samples
    samples = samples_d_xd * samples_xd

    # Compute displacement
    if percentile == -1:
        displ_meters = np.nanmean(samples)
    else:
        displ_meters = np.nanpercentile(samples, percentile * 100)

    return displ_meters