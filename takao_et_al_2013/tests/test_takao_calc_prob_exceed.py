""" """

import pytest
import sys
import numpy as np
from pathlib import Path

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from takao_et_al_2013.calc_prob_exceed import func_probex

# Test setup
RTOL = 1e-2
ATOL = 1e-4
FILE = "prob_exeed_d_ad_long_fault.csv"
FUNC = func_probex


@pytest.mark.parametrize("filename", [FILE])
def test_calc_prob_exceed_d_ad_long_fault(load_expected):

    # Inputs
    magnitude = 7
    location = 0.5
    srl_km = 100
    version = "d_ad"
    displ = load_expected["displacement"]

    # Expected
    expected = load_expected["probexceed_with_full_aleatory"]

    # Computed
    computed = FUNC(
        magnitude=magnitude,
        location=location,
        srl_km=srl_km,
        displacement_array=displ,
        version=version,
    )

    # Checks
    np.testing.assert_allclose(
        expected,
        computed,
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"For {version} long fault test, Expected: {expected}, Computed: {computed}",
    )


# Test setup
RTOL = 1e-2
ATOL = 1e-4
FILE = "prob_exeed_d_md_long_fault.csv"
FUNC = func_probex


@pytest.mark.parametrize("filename", [FILE])
def test_calc_prob_exceed_d_md_long_fault(load_expected):

    # Inputs
    magnitude = 7
    location = 0.5
    srl_km = 100
    version = "d_md"
    displ = load_expected["displacement"]

    # Expected
    expected = load_expected["probexceed_with_full_aleatory"]

    # Computed
    computed = FUNC(
        magnitude=magnitude,
        location=location,
        srl_km=srl_km,
        displacement_array=displ,
        version=version,
    )

    # Checks
    np.testing.assert_allclose(
        expected,
        computed,
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"For {version} long fault test, Expected: {expected}, Computed: {computed}",
    )


# Test setup
RTOL = 1e-2
ATOL = 1e-4
FILE = "prob_exeed_d_ad_short_fault.csv"
FUNC = func_probex


@pytest.mark.parametrize("filename", [FILE])
def test_calc_prob_exceed_d_ad_short_fault(load_expected):

    # Inputs
    magnitude = 7
    location = 0.5
    srl_km = 5
    version = "d_ad"
    displ = load_expected["displacement"]

    # Expected
    expected = load_expected["probexceed_with_full_aleatory"]

    # Computed
    computed = FUNC(
        magnitude=magnitude,
        location=location,
        srl_km=srl_km,
        displacement_array=displ,
        version=version,
    )

    # Checks
    np.testing.assert_allclose(
        expected,
        computed,
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"For {version} short fault test, Expected: {expected}, Computed: {computed}",
    )


# Test setup
RTOL = 1e-2
ATOL = 1e-4
FILE = "prob_exeed_d_md_short_fault.csv"
FUNC = func_probex


@pytest.mark.parametrize("filename", [FILE])
def test_calc_prob_exceed_d_md_short_fault(load_expected):

    # Inputs
    magnitude = 7
    location = 0.5
    srl_km = 5
    version = "d_md"
    displ = load_expected["displacement"]

    # Expected
    expected = load_expected["probexceed_with_full_aleatory"]

    # Computed
    computed = FUNC(
        magnitude=magnitude,
        location=location,
        srl_km=srl_km,
        displacement_array=displ,
        version=version,
    )

    # Checks
    np.testing.assert_allclose(
        expected,
        computed,
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"For {version} short fault test, Expected: {expected}, Computed: {computed}",
    )
