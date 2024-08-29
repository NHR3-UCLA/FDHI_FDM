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


def func_det(*, magnitude, location, percentile, version):
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

    version : str
        MEA22 normalization model name (case-insensitive). Valid options are "d_ad" or "d_md".

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

    for var in [magnitude, location, percentile]:
        if not isinstance(var, (int, float, np.integer, np.floating)):
            raise TypeError("Only one magnitude, one location, and one percentile is allowed.")

    # Calculate statistical distribution parameter predictions
    if version == "d_ad":
        mu, sigma = _calc_params_ad(magnitude=magnitude)
        alpha, beta = _calc_params_d_ad(location=location)
    elif version == "d_md":
        mu, sigma = _calc_params_md(magnitude=magnitude)
        alpha, beta = _calc_params_d_md(location=location)

    # Use sampling to convolve D_XD and XD to get total aleatory variability
    N = 500_000
    np.random.seed(1)
    samples_xd = np.power(10, stats.norm.rvs(loc=mu, scale=sigma, size=N))
    np.random.seed(1)  # this needs to be reset before each RVS
    samples_d_xd = stats.gamma.rvs(a=alpha, loc=0, scale=beta, size=N)

    # Truncation to correct for D/MD >1
    if version == "d_md":
        drop_idx = np.nonzero(samples_d_xd > 1)
        samples_d_xd[drop_idx] = np.nan

    # Combine samples
    samples = samples_d_xd * samples_xd

    # Compute displacement
    if percentile == -1:
        displ_meters = np.nanmean(samples)
    else:
        displ_meters = np.nanpercentile(samples, percentile * 100)

    return displ_meters
