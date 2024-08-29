""" """

# Python imports
import pytest
import sys
import numpy as np
from pathlib import Path

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from petersen_et_al_2011.functions import _calc_params_elliptical, _calc_params_quadratic

# Test setup
RTOL = 1e-2
FILE = "params_elliptical.csv"
FUNC = _calc_params_elliptical


@pytest.mark.parametrize("func", [FUNC])
@pytest.mark.parametrize("filename", [FILE])
def test_params_ellitical(load_expected, func, filename):
    for row in load_expected:
        # Inputs
        magnitude, location, mu_expect, sd_expect = row

        # Computed
        mu_calc, sd_calc = func(magnitude=magnitude, location=location)

        # Checks
        np.testing.assert_allclose(
            mu_expect,
            mu_calc,
            rtol=RTOL,
            err_msg=(
                f"Discrepancy in mu value for mag {magnitude} and "
                f"loc {location} for elliptical model"
            ),
        )

        np.testing.assert_allclose(
            sd_expect,
            sd_calc,
            rtol=RTOL,
            err_msg=(
                f"Discrepancy in sigma value for mag {magnitude} and "
                f"loc {location} for elliptical model"
            ),
        )


# Test setup
RTOL = 1e-2
FILE = "params_quadratic.csv"
FUNC = _calc_params_quadratic


@pytest.mark.parametrize("func", [FUNC])
@pytest.mark.parametrize("filename", [FILE])
def test_params_quadratic(load_expected, func, filename):
    for row in load_expected:
        # Inputs
        magnitude, location, mu_expect, sd_expect = row

        # Computed
        mu_calc, sd_calc = func(magnitude=magnitude, location=location)

        # Checks
        np.testing.assert_allclose(
            mu_expect,
            mu_calc,
            rtol=RTOL,
            err_msg=(
                f"Discrepancy in mu value for mag {magnitude} and "
                f"loc {location} for quadratic model"
            ),
        )

        np.testing.assert_allclose(
            sd_expect,
            sd_calc,
            rtol=RTOL,
            err_msg=(
                f"Discrepancy in sigma value for mag {magnitude} and "
                f"loc {location} for quadratic model"
            ),
        )
