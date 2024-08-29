"""Functions to calculate the model predictions."""

# Python imports
import sys
import numpy as np

from pathlib import Path

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chiou_et_al_2024.calc_deterministic import func_det


def func_profile(*, magnitude, percentile, location_step=0.05, version="model7"):
    """
    Calculate the predicted displacement profile in meters.

    Parameters
    ----------
    magnitude : float
        The earthquake moment magnitude.

    percentile : int or float or numpy.ndarray with a single element
        Aleatory quantile value. Use -1 for mean.

    location_step : float, optional
        Profile location step interval. Default 0.05.

    version : str
        Specify which Chiou et al. nEMG model to use (case-insensitive). The default is `model7`.
        Valid options are `model7`, `model8_1`, `model8_2`, or `model8_3`.
        Only one value is allowed.

    Returns
    -------
    tuple
        - 'locations': Normalized location along rupture length.
        - 'displ_meters': Displacement in meters.
    """

    locations = np.arange(0, 1 + location_step, location_step)
    displ_meters = func_det(magnitude=magnitude, location=locations, percentile=percentile)

    return locations, displ_meters
