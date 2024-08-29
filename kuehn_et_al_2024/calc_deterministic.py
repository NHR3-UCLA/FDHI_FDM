"""Functions to calculate the model predictions."""

# Python imports
import sys
import numpy as np
import pandas as pd

from pathlib import Path

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kuehn_et_al_2024.functions import _calc_params, _calc_transformed_displ


def func_det(
    *, magnitude, location, style, percentile, coefficient_type="median", folded=True, debug=False
):
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

    coefficient_type : str, optional
        Option to run model using full epistemic uncertainty or with point estimates (mean or
        median) of the model coefficients (case-insensitive). Valid options are 'mean', 'median',
        or 'full'. Default 'median'.

    folded : boolean, optional
        Return displacement for the folded location. Default True.

    debug : boolean, optional
        Option to return DataFrame of internal calculations. Default False.

    Returns
    -------
    If debug is False:
        displ_folded_meters : numpy.ndarray
            Displacement in meters for the folded location. The array contains a single element.

    If debug is True:
        pd.DataFrame
            A DataFrame with the following columns:

            - **magnitude**: Earthquake moment magnitude.
            - **location**: Normalized location along rupture length.
            - **style**: Style of faulting.
            - **percentile**: The percentile for which the displacement is calculated.
            - **model_id**: Model coefficient row number or point estimate definition.
            - **bc_param**: Box-Cox transformation parameter (lambda).
            - **mean_site**: Mean displacement in transformed units for the site location.
            - **stdv_site**: Total standard deviation in transformed units for the site location.
            - **mean_complement**: Mean displacement in transformed units for the complementary location.
            - **stdv_complement**: Total standard deviation in transformed units for the complementary location.
            - **Y_site**: Transformed displacement for the site location.
            - **Y_complement**: Transformed displacement for the complementary location.
            - **Y_folded**: Transformed displacement for the folded location.
            - **displ_site_meters**: Displacement in meters for the site location.
            - **displ_complement_meters**: Displacement in meters for the complementary location.
            - **displ_folded_meters**: Displacement in meters for the folded location.
    """

    # Calculate statistical distribution parameter predictions
    coefficient_type = coefficient_type.lower()
    params = {"magnitude": magnitude, "style": style, "coefficient_type": coefficient_type}
    model_id, bc_param, mean_site, stdv_site, _, _ = _calc_params(**params, location=location)
    _, _, mean_complement, stdv_complement, _, _ = _calc_params(**params, location=1 - location)

    # Calculate transformed displacement
    Y_site = _calc_transformed_displ(bc_param, mean_site, stdv_site, percentile)
    Y_complement = _calc_transformed_displ(bc_param, mean_complement, stdv_complement, percentile)
    Y_folded = np.mean([Y_site, Y_complement], axis=0)

    # Back-transform displacement to meters
    displ_site_meters = np.power(Y_site * bc_param + 1, 1 / bc_param)
    displ_complement_meters = np.power(Y_complement * bc_param + 1, 1 / bc_param)  # noqa: F841
    displ_folded_meters = np.power(Y_folded * bc_param + 1, 1 / bc_param)

    if debug:
        result = {
            k: v
            for k, v in locals().items()
            if k not in ["coefficient_type", "folded", "debug", "params", "_"]
        }
        return pd.DataFrame.from_dict(result)
    else:
        return displ_folded_meters if folded else displ_site_meters
