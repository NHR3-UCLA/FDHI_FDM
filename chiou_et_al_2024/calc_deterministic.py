"""Functions to calculate the model predictions."""

# Python imports
import sys
import numpy as np

from pathlib import Path
from scipy.stats import exponnorm

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chiou_et_al_2024.functions import _calc_params


def func_det(*, magnitude, location, percentile, version="model7"):
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
        Specify which Chiou et al. nEMG model to use (case-insensitive). The default is `model7`.
        Valid options are `model7`, `model8_1`, `model8_2`, or `model8_3`.
        Only one value is allowed.

    Returns
    -------
    float
        The displacement in meters.
    """

    # User input checks
    version = version.lower()
    valid_versions = {"model7", "model8_1", "model8_2", "model8_3"}
    if version not in valid_versions:
        raise ValueError(
            f"Invalid version. Please choose one of the valid options: {valid_versions}."
        )

    # Calculate statistical distribution parameter predictions
    mu, nu, sigma_prime, _, _ = _calc_params(magnitude=magnitude, l2L=location, version=version)

    # Compute displacement
    if percentile == -1:
        D = np.exp(mu + np.power(sigma_prime, 2) / 2) / (nu + 1)
    else:
        # Note scipy parametrization of shape parameter is different than R gamless
        shape = 1 / (sigma_prime * (1 / nu))
        ln_D = -1 * exponnorm.ppf(q=1 - percentile, K=shape, loc=-mu, scale=sigma_prime)
        D = np.exp(ln_D)

    return D
