"""Functions to calculate the model predictions."""

# Python imports
import sys
import numpy as np

from pathlib import Path
from scipy import stats

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from petersen_et_al_2011.functions import _calc_params_elliptical, _calc_params_quadratic


def func_probex(*, magnitude, location, displacement_array, version="model7"):
    """
    Calculate the probability of exceedance for a scenario and array of displacement test values.

    Parameters
    ----------
    magnitude : float
        The earthquake moment magnitude.

    location : int or float
        Normalized location along rupture length, range [0, 1.0].

    displacement_array : array-like
        Displacement test values in meters.

    version : str
        Specify which Petersent et al. profile shape model to use (case-insensitive). The default
        is `elliptical`. Valid options are `elliptical` or `quadratic`. Only one value is allowed.
        Only one value is allowed.

    Returns
    -------
    array-like
        The probability of exceedance.
    """

    # User input checks
    version = version.lower()
    valid_versions = {"elliptical", "quadratic"}
    if version not in valid_versions:
        raise ValueError(
            f"Invalid version. Please choose one of the valid options: {valid_versions}."
        )

    for var in [magnitude, location]:
        if not isinstance(var, (int, float, np.integer, np.floating)):
            raise TypeError("Only one magnitude and one location is allowed.")

    # Calculate statistical distribution parameter predictions
    function_map = {"elliptical": _calc_params_elliptical, "quadratic": _calc_params_quadratic}
    mu, sd = function_map[version](magnitude=magnitude, location=location)

    # Calculate probabilities of exceedance
    z = np.log(np.multiply(displacement_array, 100))
    ccdf = 1 - stats.norm.cdf(x=z, loc=mu, scale=sd)

    return ccdf
