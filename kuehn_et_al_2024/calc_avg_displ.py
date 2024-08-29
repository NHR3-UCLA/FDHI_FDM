"""Functions to calculate the model predictions."""

# Python imports
import sys
import numpy as np

from pathlib import Path

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kuehn_et_al_2024.functions import _calc_params, _calc_analytic_mean


def func_ad(*, magnitude, style, coefficient_type="median"):
    """
    Calculate the average displacement in meters.

    Parameters
    ----------
    magnitude : int or float or numpy.ndarray with a single element
        Earthquake moment magnitude.

    style : str
        Style of faulting (case-insensitive). Valid options are 'strike-slip', 'reverse', or
        'normal'.

    coefficient_type : str, optional
        Option to run model using mean or median point estimates of the model coefficients (case-
        insensitive). Valid options are 'mean' or 'median'. (The 'full' option is not enabled for
        this function.) Default 'median'.

    Returns
    -------
    float
        Average displacement in meters.
    """

    coefficient_type = coefficient_type.lower()
    if coefficient_type not in ["mean", "median"]:
        raise ValueError(
            f"'{coefficient_type}' is an invalid 'coefficient_type';"
            " only 'mean' or 'median' is allowed."
        )

    # Calculate statistical distribution parameter predictions
    # Dense location spacing is used to create well-descritized profile for intergration
    params = {"magnitude": magnitude, "style": style, "coefficient_type": coefficient_type}
    locations = np.arange(0, 1.01, 0.01)
    results = np.array([_calc_params(**params, location=l) for l in locations])
    results = np.squeeze(results)
    _, bc_param, mean, _, stdv_within, _ = results.T

    # Calculate predicted mean slip profile
    # Use within-event variability only for median AD; see manucript for discussion
    mean_displ_meters = _calc_analytic_mean(bc_param, mean, stdv_within)

    # Calculate area under the mean slip profile; this is the Average Displacement (AD)
    return np.trapz(mean_displ_meters, locations)
