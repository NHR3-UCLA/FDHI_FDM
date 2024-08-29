"""Functions to calculate the model predictions."""

# Python imports
import sys
import numpy as np

from pathlib import Path
from scipy import stats

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wells_coppersmith_1994.functions import _calc_params_md


def func_md(*, magnitude, percentile, style="all"):
    """
    Calculate the maximum displacement in meters.

    Parameters
    ----------
    magnitude : array-like
        The earthquake moment magnitude.

    percentile : int or float or numpy.ndarray with a single element
        Aleatory quantile value. Use -1 for mean.

    style : str, optional
        Style of faulting (case-insensitive). Default is "all". Valid options are "strike-slip",
        "reverse", "normal", or "all".

    Returns
    -------
    array-like
        The maximum displacement in meters.
    """

    if not isinstance(percentile, (int, float, np.integer, np.floating)):
        raise TypeError("Only one percentile is allowed.")

    mu, sigma = _calc_params_md(magnitude=magnitude, style=style)

    if percentile == -1:
        log10_displ = mu + (np.log(10) / 2 * np.power(sigma, 2))
    else:
        log10_displ = stats.norm.ppf(percentile, loc=mu, scale=sigma)

    return np.power(10, log10_displ)
