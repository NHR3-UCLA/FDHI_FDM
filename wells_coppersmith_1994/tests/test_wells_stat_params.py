""" """

# Python imports
import pytest
import sys
import numpy as np
from pathlib import Path

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wells_coppersmith_1994.functions import _calc_params_ad, _calc_params_md

# Test setup
RTOL = 1e-2
FILE = "params_mag_ad.csv"
FUNC = _calc_params_ad


@pytest.mark.parametrize("func", [FUNC])
@pytest.mark.parametrize("filename", [FILE])
def test_params_ad(load_expected, func, filename):
    for row in load_expected:
        # Inputs
        magnitude, style, mu_expect, sd_expect = row

        # Computed
        mu_calc, sd_calc = func(magnitude=magnitude, style=style)

        # Checks
        np.testing.assert_allclose(
            mu_expect,
            mu_calc,
            rtol=RTOL,
            err_msg=(f"Discrepancy in mu value for {style} mag {magnitude}"),
        )

        np.testing.assert_allclose(
            sd_expect,
            sd_calc,
            rtol=RTOL,
            err_msg=(f"Discrepancy in sigma value for {style} mag {magnitude}"),
        )


# Test setup
RTOL = 1e-2
FILE = "params_mag_md.csv"
FUNC = _calc_params_md


@pytest.mark.parametrize("func", [FUNC])
@pytest.mark.parametrize("filename", [FILE])
def test_params_md(load_expected, func, filename):
    for row in load_expected:
        # Inputs
        magnitude, style, mu_expect, sd_expect = row

        # Computed
        mu_calc, sd_calc = func(magnitude=magnitude, style=style)

        # Checks
        np.testing.assert_allclose(
            mu_expect,
            mu_calc,
            rtol=RTOL,
            err_msg=(f"Discrepancy in mu value for {style} mag {magnitude}"),
        )

        np.testing.assert_allclose(
            sd_expect,
            sd_calc,
            rtol=RTOL,
            err_msg=(f"Discrepancy in sigma value for {style} mag {magnitude}"),
        )
