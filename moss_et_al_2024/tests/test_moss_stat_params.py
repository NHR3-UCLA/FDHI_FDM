""" """

# Python imports
import pytest
import sys
import numpy as np
from pathlib import Path

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from moss_et_al_2024.functions import (
    _calc_params_ad,
    _calc_params_md,
    _calc_params_d_ad,
    _calc_params_d_md,
)

# Test setup
RTOL = 1e-2
FILE = "params_mag_ad.csv"
FUNC = _calc_params_ad


@pytest.mark.parametrize("func", [FUNC])
@pytest.mark.parametrize("filename", [FILE])
def test_params_ad(load_expected, func, filename):
    # Define inputs and expected outputs
    magnitude = load_expected["magnitude"]
    mu_expected = load_expected["mu"]
    sigma_expected = load_expected["sigma"]

    # Perform calculations
    mu_calculated, sigma_calculated = func(magnitude=magnitude)

    # Assert expected vs. calculated values
    np.testing.assert_allclose(
        mu_expected, mu_calculated, rtol=RTOL, err_msg="Discrepancy in mu values"
    )
    np.testing.assert_allclose(
        sigma_expected, sigma_calculated, rtol=RTOL, err_msg="Discrepancy in sigma values"
    )


# Test setup
FILE = "params_mag_md.csv"
FUNC = _calc_params_md


@pytest.mark.parametrize("func", [FUNC])
@pytest.mark.parametrize("filename", [FILE])
def test_params_md(load_expected, func, filename):
    # Define inputs and expected outputs
    magnitude = load_expected["magnitude"]
    mu_expected = load_expected["mu"]
    sigma_expected = load_expected["sigma"]

    # Perform calculations
    mu_calculated, sigma_calculated = func(magnitude=magnitude)

    # Assert expected vs. calculated values
    np.testing.assert_allclose(
        mu_expected, mu_calculated, rtol=RTOL, err_msg="Discrepancy in mu values"
    )
    np.testing.assert_allclose(
        sigma_expected, sigma_calculated, rtol=RTOL, err_msg="Discrepancy in sigma values"
    )


# Test setup
FILE = "params_d_ad.csv"
FUNC = _calc_params_d_ad


@pytest.mark.parametrize("func", [FUNC])
@pytest.mark.parametrize("filename", [FILE])
def test_params_d_ad(load_expected, func, filename):
    # Define inputs and expected outputs
    location = load_expected["location"]
    alpha_expect = load_expected["alpha"]
    beta_expect = load_expected["beta"]

    # Perform calculations
    results = func(location=location)
    alpha_calc, beta_calc = results

    # Comparing exepcted and calculated
    np.testing.assert_allclose(
        alpha_expect, alpha_calc, rtol=RTOL, err_msg="Discrepancy in alpha values"
    )
    np.testing.assert_allclose(
        beta_expect, beta_calc, rtol=RTOL, err_msg="Discrepancy in beta values"
    )


# Test setup
FILE = "params_d_md.csv"
FUNC = _calc_params_d_md


@pytest.mark.parametrize("func", [FUNC])
@pytest.mark.parametrize("filename", [FILE])
def test_params_d_md(load_expected, func, filename):
    # Define inputs and expected outputs
    location = load_expected["location"]
    alpha_expect = load_expected["alpha"]
    beta_expect = load_expected["beta"]

    # Perform calculations
    results = func(location=location)
    alpha_calc, beta_calc = results

    # Comparing exepcted and calculated
    np.testing.assert_allclose(
        alpha_expect, alpha_calc, rtol=RTOL, err_msg="Discrepancy in alpha values"
    )
    np.testing.assert_allclose(
        beta_expect, beta_calc, rtol=RTOL, err_msg="Discrepancy in beta values"
    )
