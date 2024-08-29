"""Functions to calculate the model predictions."""

# Python imports
import sys
import numpy as np
from pathlib import Path

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kuehn_et_al_2024.calc_deterministic import func_det


def func_profile(
    *, magnitude, style, percentile, coefficient_type="median", folded=True, location_step=0.05
):
    """
    Calculate the predicted displacement profile in meters.

    Parameters
    ----------
    magnitude : int or float or numpy.ndarray with a single element
        Earthquake moment magnitude.

    style : str
        Style of faulting (case-insensitive). Valid options are 'strike-slip', 'reverse', or
        'normal'.

    percentile : int or float or numpy.ndarray with a single element
        Aleatory quantile value. Use -1 for mean.

    coefficient_type : str, optional
        Option to run model using mean or median point estimates of the model coefficients (case-
        insensitive). Valid options are 'mean' or 'median'. (The 'full' option is not enabled for
        this function.) Default 'median'.

    folded : boolean, optional
        Return displacement for the folded location. Default True.

    location_step : float, optional
        Profile location step interval. Default 0.05.

    Returns
    -------
    tuple
        - 'locations': Normalized location along rupture length.
        - 'displ_meters': Displacement in meters.
    """

    coefficient_type = coefficient_type.lower()
    if coefficient_type not in ["mean", "median"]:
        raise ValueError(
            f"'{coefficient_type}' is an invalid 'coefficient_type';"
            " only 'mean' or 'median' is allowed for the profile."
        )

    locations = np.arange(0, 1 + location_step, location_step)

    displ_meters = np.array(
        [
            func_det(
                magnitude=magnitude,
                location=l,
                style=style,
                percentile=percentile,
                coefficient_type=coefficient_type,
                folded=folded,
            )
            for l in locations
        ]
    )

    return locations, displ_meters
