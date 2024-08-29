"""Functions to calculate the model predictions."""

# Python imports
import sys

from pathlib import Path
from scipy import stats as scipystats

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lavrentiadis_abrahamson_2023.functions import LavrentiadisAbrahamson2023MaxDisp, _calc_mean


def func_md(*, magnitude, style, percentile):
    """
    Calculate the maximum displacement in meters.

    Parameters
    ----------
    magnitude : int or float or numpy.ndarray with a single element
        Earthquake moment magnitude.

    style : str
        Style of faulting (case-insensitive). Valid options are 'strike-slip', 'reverse', or
        'normal'.

    percentile : int or float or numpy.ndarray with a single element
        Aleatory quantile value. Use -1 for mean.

    Returns
    -------
    int or float or numpy.ndarray with a single element
        Maximum aggregate displacement in meters.
    """

    # Note SRL is just used for normalization in the script, so it can be set to 1
    median_md, sd = LavrentiadisAbrahamson2023MaxDisp(magnitude, 1, style)

    if percentile == -1:
        return _calc_mean(median_md ** (0.3), sd)
    else:
        return scipystats.norm.ppf(percentile, loc=median_md ** (0.3), scale=sd) ** (10 / 3)
