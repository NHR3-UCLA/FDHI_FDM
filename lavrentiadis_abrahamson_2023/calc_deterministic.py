"""Functions to calculate the model predictions."""

# Python imports
import sys
import numpy as np

from pathlib import Path

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lavrentiadis_abrahamson_2023.functions import (
    LavrentiadisAbrahamson2023SlipProfile,
    LavrentiadisAbrahamson2023SlipProfilePrc,
    _calc_mean,
)


def func_det(*, magnitude, location, style, percentile):
    """
    Calculate the displacement in meters for a scenario.

    Parameters
    ----------
    magnitude : int or float or numpy.ndarray with a single element
        Earthquake moment magnitude.

    location : int or float or numpy.ndarray with a single element
        Normalized location along rupture length, range [0, 1.0].

    style : str
        Style of faulting (case-insensitive). Valid options are 'strike-slip', 'reverse', or
        'normal'.

    percentile : int or float or numpy.ndarray with a single element
        Aleatory quantile value. Use -1 for mean.

    Returns
    -------
    tuple
        - 'disp_agg_prime': Full-rupture aggregate displacement in meters.
        - 'disp_prnc_prime': Full-rupture principal displacement in meters.
        - 'disp_agg_seg': Single-segment aggregate displacement in meters.
        - 'P_gap': Segment gap probability.
        - 'P_zero_slip': Zero principal slip probability.
    """

    # Note SRL is just used for normalization in the script, so it can be set to 1

    (
        disp_agg_prime_q50,
        disp_prnc_prime_q50,
        disp_agg_seg_q50,
        sig_agg,
        sig_prnc,
        phi_agg,
        phi_prnc,
        tau_agg,
        phi_add,
        P_gap,
        P_zero_slip,
    ) = LavrentiadisAbrahamson2023SlipProfile(location, magnitude, 1, style)

    if percentile == -1:
        disp_agg_prime = _calc_mean(disp_agg_prime_q50 ** (0.3), sig_agg)
        disp_prnc_prime = _calc_mean(disp_prnc_prime_q50 ** (0.3), sig_prnc)

        sig_agg_seg = np.sqrt(tau_agg**2 + phi_agg**2)
        disp_agg_seg = _calc_mean(disp_agg_seg_q50 ** (0.3), sig_agg_seg)
    else:
        disp_agg_prime, disp_prnc_prime, disp_agg_seg = LavrentiadisAbrahamson2023SlipProfilePrc(
            location, magnitude, 1, style, percentile
        )

    return disp_agg_prime, disp_prnc_prime, disp_agg_seg, P_gap, P_zero_slip
