"""Functions to calculate the model predictions."""

# Python imports
import sys
import numpy as np

from pathlib import Path

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from petersen_et_al_2011.calc_profile import func_profile


def func_ad(*, magnitude, version="elliptical"):
    """
    Calculate the mean prediction for average displacement in meters. Note that the median
    prediction cannot be calculated because the aleatory variability is not partitioned in this
    model.

    Parameters
    ----------
    magnitude : int or float or numpy.ndarray with a single element
        Earthquake moment magnitude.

    version : str
        Specify which Petersent et al. profile shape model to use (case-insensitive). The default
        is `elliptical`. Valid options are `elliptical` or `quadratic`. Only one value is allowed.

    Returns
    -------
    float
        Mean predition for average displacement in meters.
    """

    # Calculate area under the mean slip profile; this is the Average Displacement (AD)
    # It includes within-event and between-event variability
    # Dense location spacing is used to create well-descritized profile for intergration
    locations, mean_displ_meters = func_profile(
        magnitude=magnitude, percentile=-1, version=version, location_step=0.01
    )

    return np.trapz(mean_displ_meters, locations)
