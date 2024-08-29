"""Functions to calculate the model predictions."""

# Python imports
import sys
import numpy as np

from pathlib import Path
from scipy import stats

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from petersen_et_al_2011.functions import _calc_params_elliptical, _calc_params_quadratic


def func_det(*, magnitude, location, percentile, version="elliptical"):
    """
    Calculate the displacement in meters for a scenario.

    Parameters
    ----------
    magnitude : float
        The earthquake moment magnitude.

    location : int or float
        Normalized location along rupture length, range [0, 1.0].

    percentile : float
        The aleatory quantile in the range [0,1]. Use -1 for mean.

    version : str
        Specify which Petersent et al. profile shape model to use (case-insensitive). The default
        is `elliptical`. Valid options are `elliptical` or `quadratic`. Only one value is allowed.

    Returns
    -------
    float
        The displacement in meters.
    """

    # User input checks
    version = version.lower()
    valid_versions = {"elliptical", "quadratic"}
    if version not in valid_versions:
        raise ValueError(
            f"Invalid version. Please choose one of the valid options: {valid_versions}."
        )

    # Calculate statistical distribution parameter predictions
    function_map = {"elliptical": _calc_params_elliptical, "quadratic": _calc_params_quadratic}
    mu, sd = function_map[version](magnitude=magnitude, location=location)

    # Compute displacement
    if percentile == -1:
        ln_displ_cm = mu + np.power(sd, 2) / 2
    else:
        ln_displ_cm = stats.norm.ppf(percentile, loc=mu, scale=sd)

    return np.exp(ln_displ_cm) / 100
