"""Functions to calculate the model predictions."""

# Python imports
import sys
import numpy as np

from pathlib import Path
from scipy import stats as scipystats

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lavrentiadis_abrahamson_2023.functions import LavrentiadisAbrahamson2023SlipProfile


def func_probex(*, magnitude, location, style, displacement_array):
    """
    Calculate the probability of exceedance for a scenario and array of displacement test values.

    Parameters
    ----------
    magnitude : int or float or numpy.ndarray with a single element
        Earthquake moment magnitude.

    location : int or float or numpy.ndarray with a single element
        Normalized location along rupture length, range [0, 1.0].

    style : str
        Style of faulting (case-insensitive). Valid options are 'strike-slip', 'reverse', or
        'normal'.

    displacement_array : array-like
        Displacement test values in meters.

    Returns
    -------
    tuple
        - 'ccdf_agg': Probability of exceedance for full-rupture nonzero aggregate displacement.
        - 'ccdf_prnc': Probability of exceedance for full-rupture nonzero principal displacement.
        - 'ccdf_seg': Probability of exceedance for single-segment nonzero aggregate displacement.
        - 'ccdf_agg_with_zero': Probability of exceedance for full-rupture aggregate displacement with P_gap.
        - 'ccdf_agg_with_zero': Probability of exceedance for full-rupture nonzero principal displacement with P_gap and P_zero_slip.
        - 'ccdf_agg_with_zero': Probability of exceedance for single-segment nonzero aggregate displacement with P_zero_slip.
    """

    # Note SRL is just used for normalization in the script, so it can be set to 1

    for var in [magnitude, location]:
        if not isinstance(var, (int, float, np.integer, np.floating)):
            raise TypeError("Only one magnitude and one location is allowed.")

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
    ) = LavrentiadisAbrahamson2023SlipProfile(location, magnitude, 1, style)

    mu_agg_p = disp_agg_prime**0.3
    mu_prnc_p = disp_prnc_prime**0.3

    mu_agg_seg = disp_agg_seg**0.3
    sig_agg_seg = np.sqrt(tau_agg**2 + phi_agg**2)

    z = np.atleast_1d(displacement_array) ** 0.3

    ccdf_agg = scipystats.norm.sf(x=z, loc=mu_agg_p, scale=sig_agg)
    ccdf_prnc = scipystats.norm.sf(x=z, loc=mu_prnc_p, scale=sig_prnc)
    ccdf_seg = scipystats.norm.sf(x=z, loc=mu_agg_seg, scale=sig_agg_seg)

    ccdf_agg_with_zero = ccdf_agg * (1 - P_gap)
    ccdf_prnc_with_zero = ccdf_prnc * (1 - P_zero_slip) * (1 - P_gap)
    ccdf_seg_with_zero = ccdf_seg * (1 - P_zero_slip)

    return (
        ccdf_agg,
        ccdf_prnc,
        ccdf_seg,
        ccdf_agg_with_zero,
        ccdf_prnc_with_zero,
        ccdf_seg_with_zero,
    )
