"""Functions to calculate the model predictions."""

# Python imports
import sys
import numpy as np
from pathlib import Path

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lavrentiadis_abrahamson_2023.calc_deterministic import func_det


def func_profile(*, magnitude, style, percentile, location_step=0.05):
    """
    Calculate the predicted displacement profile in meters.

    Parameters
    ----------
    magnitude : int or float or numpy.ndarray with a single element
        Earthquake moment magnitude.

    style : str
        Style of faulting (case-insensitive). Valid options are 'strike-slip', 'reverse', or
        'normal'.

    location_step : float, optional
        Profile location step interval. Default 0.05.

    Returns
    -------
    tuple
        - 'locations': Normalized location along rupture length.
        - 'disp_agg_p': Full-rupture aggregate displacement in meters.
        - 'disp_prnc_p': Full-rupture principal displacement in meters.
        - 'disp_agg_seg': Single-segment aggregate displacement in meters.
        - 'P_gap': Segment gap probability.
        - 'P_zero_slip': Zero principal slip probability.
    """

    for var in [magnitude, percentile]:
        if not isinstance(var, (int, float, np.integer, np.floating)):
            raise TypeError("Only one magnitude and one percentile is allowed.")

    locations = np.arange(0, 1 + location_step, location_step)

    results = np.array(
        [
            func_det(magnitude=magnitude, location=l, style=style, percentile=percentile)
            for l in locations
        ]
    )
    results = np.squeeze(results)
    disp_agg_p, disp_prnc_p, disp_agg_seg, P_gap, P_zero_slip = results.T

    return locations, disp_agg_p, disp_prnc_p, disp_agg_seg, P_gap, P_zero_slip
