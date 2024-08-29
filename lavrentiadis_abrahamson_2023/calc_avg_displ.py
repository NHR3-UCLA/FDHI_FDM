"""Functions to calculate the model predictions."""

# Python imports
import sys
import numpy as np

from pathlib import Path

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lavrentiadis_abrahamson_2023.functions import (
    LavrentiadisAbrahamson2023AvgDisp,
    LavrentiadisAbrahamson2023SlipProfile,
    _calc_mean,
)


def func_ad(*, magnitude, style):
    """
    Calculate the average displacement in meters.

    Parameters
    ----------
    magnitude : array-like
        Earthquake moment magnitude.

    style : str
        Style of faulting (case-insensitive). Valid options are 'strike-slip', 'reverse', or
        'normal'.

    Returns
    -------
    array-like
        Average principal displacement in meters.
    """

    # Note SRL is just used for normalization in the script, so it can be set to 1
    return LavrentiadisAbrahamson2023AvgDisp(np.atleast_1d(magnitude), 1, style)[0]


def _func_integrate_for_ad(*, magnitude, style):
    """
    Compute average displacement through integration; i.e., solve Eqn. 25 in LA23.

    Parameters
    ----------
    magnitude : int or float or numpy.ndarray with a single element
        Earthquake moment magnitude.

    style : str
        Style of faulting (case-insensitive). Valid options are 'strike-slip', 'reverse', or
        'normal'.

    Returns
    -------
    tuple
        - 'ad_agg_prime': Average aggregate displacement in meters.
        - 'ad_prnc_prime': Average principal displacement in meters.
    """

    # Note SRL is just used for normalization in the script, so it can be set to 1

    locations = np.arange(0, 0.51, 0.01)

    (
        disp_agg_prime,
        disp_prnc_prime,
        disp_agg_seg,
        sig_agg,
        sig_prnc,
        phi_agg,
        phi_prnc,
        tau_agg,
        phi_add,
        P_gap,
        P_zero_slip,
    ) = LavrentiadisAbrahamson2023SlipProfile(locations, magnitude, 1, style)

    def _calc_area(mean_pn, sd_pn, x_array=locations):
        half_profile = _calc_mean(mean_pn, sd_pn)
        return np.trapz(half_profile, x_array) * 2

    # Calculate average displacement as area under mean profile with only within-event variability
    ad_agg_prime = _calc_area(disp_agg_prime ** (0.3), phi_agg)
    ad_prnc_prime = _calc_area(disp_prnc_prime ** (0.3), phi_prnc)

    return ad_agg_prime, ad_prnc_prime
