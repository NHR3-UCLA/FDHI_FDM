"""Functions to calculate the model predictions."""

# Python imports
import sys
import numpy as np

from pathlib import Path

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from takao_et_al_2013.calc_deterministic import func_det


def func_profile(*, magnitude, percentile, srl_km, version, location_step=0.05):
    """
    Calculate the predicted displacement profile in meters.

    Parameters
    ----------
    magnitude : float
        The earthquake moment magnitude.

    percentile : int or float or numpy.ndarray with a single element
        Aleatory quantile value. Use -1 for mean.

    srl_km : int or float
        Total surface rupture length in kilometers.

    version : str
        TEA13 normalization model name (case-insensitive). Valid options are "d_ad" or "d_md".

    location_step : float, optional
        Profile location step interval. Default 0.05.

    Returns
    -------
    tuple
        - 'locations': Normalized location along rupture length.
        - 'displ_meters': Displacement in meters.
    """

    # User input checks
    valid_versions = {"d_ad", "d_md"}
    if version.lower() not in valid_versions:
        raise ValueError(
            f"Invalid version. Please choose one of the valid options: {valid_versions}."
        )

    for var in [magnitude, percentile, srl_km]:
        if not isinstance(var, (int, float, np.integer, np.floating)):
            raise TypeError("Only one magnitude, one SRL, and one percentile is allowed.")

    locations = np.arange(0, 1 + location_step, location_step)

    # Note: results D_XD and XD are convolved to include total aleatory variability
    displ_meters = np.array(
        [
            func_det(
                magnitude=magnitude,
                location=l,
                percentile=percentile,
                srl_km=srl_km,
                version=version,
            )
            for l in locations
        ]
    )

    return locations, displ_meters
