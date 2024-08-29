""" """

import pytest
import sys
import numpy as np
from pathlib import Path

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kuehn_et_al_2024.load_data import DATA
from kuehn_et_al_2024.functions import _func_ss

# Test setup
RTOL = 1e-2
STYLE = "strike-slip"
MODEL = "mean"
FILE = "strike-slip_mean_coeffs.csv"
FUNC = _func_ss


@pytest.mark.parametrize("filename", [FILE])
def test_func_ss(load_expected):

    # Coefficients
    coeffs = DATA["point"][STYLE]
    coeffs = coeffs[coeffs["model_id"] == MODEL].to_records(index=False)

    # Inputs
    magnitudes = load_expected["mag"]
    locations = load_expected["u_star"]

    # Expected
    expected_values = {
        "median": load_expected["mu"],
        "sd_mode": load_expected["sd_mode"],
        "sd_u": load_expected["sd_u"],
        "sd_total": load_expected["sd_tot"],
    }

    # Computed
    computed = FUNC(coeffs, magnitudes, locations)
    computed_values = {
        "median": computed[2],
        "sd_mode": computed[5],
        "sd_u": computed[4],
        "sd_total": computed[3],
    }

    # Checks
    for key, expected in expected_values.items():
        computed = computed_values[key]
        np.testing.assert_allclose(
            expected,
            computed,
            rtol=RTOL,
            err_msg=f"Mag {magnitudes}, u-star {locations}, {key}, Expected: {expected}, Computed: {computed}",
        )
