"""Functions to calculate the model predictions."""

# Python imports
import sys
import numpy as np

from pathlib import Path
from scipy import stats

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from takao_et_al_2013.functions import _calc_params_md


def func_md(*, magnitude, percentile):
    """
    Calculate the maximum displacement in meters.

    Parameters
    ----------
    magnitude : array-like
        The earthquake moment magnitude.

    percentile : int or float or numpy.ndarray with a single element
        Aleatory quantile value. Use -1 for mean.

    Returns
    -------
    array-like
        The maximum displacement in meters.
    """

    if not isinstance(percentile, (int, float, np.integer, np.floating)):
        raise TypeError("Only one percentile is allowed.")

    mu, sigma = _calc_params_md(magnitude=magnitude)

    if percentile == -1:
        log10_displ = mu + (np.log(10) / 2 * np.power(sigma, 2))
    else:
        log10_displ = stats.norm.ppf(percentile, loc=mu, scale=sigma)

    return np.power(10, log10_displ)
