"""Functions to calculate the model predictions."""

# Python imports
import sys
import numpy as np

from pathlib import Path
from scipy.stats import exponnorm

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chiou_et_al_2024.functions import _calc_params


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
        Specify which Chiou et al. nEMG model to use (case-insensitive). The default is `model7`.
        Valid options are `model7`, `model8_1`, `model8_2`, or `model8_3`.
        Only one value is allowed.

    Returns
    -------
    array-like
        The probability of exceedance.
    """

    # User input checks
    version = version.lower()
    valid_versions = {"model7", "model8_1", "model8_2", "model8_3"}
    if version not in valid_versions:
        raise ValueError(
            f"Invalid version. Please choose one of the valid options: {valid_versions}."
        )

    for var in [magnitude, location]:
        if not isinstance(var, (int, float, np.integer, np.floating)):
            raise TypeError("Only one magnitude and one location is allowed.")

    # Calculate statistical distribution parameter predictions
    mu, nu, sigma_prime, _, _ = _calc_params(magnitude=magnitude, l2L=location, version=version)

    z = np.log(displacement_array)

    # Note scipy parametrization of shape parameter is different than R gamless
    # Note negative modification to EMG distribution "flips" cdf & ccdf
    shape = 1 / (sigma_prime * (1 / nu))
    ccdf = exponnorm.cdf(x=z * -1, K=shape, loc=-mu, scale=sigma_prime)

    return ccdf
