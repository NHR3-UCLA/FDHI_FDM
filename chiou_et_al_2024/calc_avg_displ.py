"""Functions to calculate the model predictions."""

# Python imports
import sys
import numpy as np

from pathlib import Path

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chiou_et_al_2024.functions import _func_mag


def func_ad(*, magnitude, version="model7"):
    """
    Calculate the average displacement in meters using the 1/2.5512 scaling.

    Parameters
    ----------
    magnitude : array-like
        The earthquake moment magnitude.

    version : str
        Specify which Chiou et al. nEMG model to use (case-insensitive). The default is `model7`.
        Valid options are `model7`, `model8_1`, `model8_2`, or `model8_3`.
        Only one value is allowed.

    Returns
    -------
    array-like
        The average displacement in meters.
    """

    coeffs_c0 = {"model7": 1.33228, "model8_1": 1.82276, "model8_2": 0.45201, "model8_3": -0.56569}

    f_m = _func_mag(magnitude=magnitude)
    c0 = coeffs_c0[version]

    AD = np.exp(c0 + f_m) * 0.3920

    return AD
