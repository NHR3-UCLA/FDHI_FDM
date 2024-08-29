""" """

import pytest
import sys
import numpy as np
from pathlib import Path

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from moss_et_al_2024.calc_prob_exceed import func_probex

# Test setup
RTOL = 1e-2
ATOL = 1e-4
FILE = "prob_exeed_d_ad.csv"
FUNC = func_probex


@pytest.mark.parametrize("filename", [FILE])
def test_calc_prob_exceed_d_ad(load_expected):

    # Inputs
    magnitude = 7
    location = 0.5
    version = "d_ad"
    displ = load_expected["displacement"]

    # Expected
    expected = load_expected["probexceed_with_full_aleatory"]

    # Computed
    computed = FUNC(
        magnitude=magnitude,
        location=location,
        displacement_array=displ,
        version=version,
    )

    # Checks
    np.testing.assert_allclose(
        expected,
        computed,
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"For {version} test, Expected: {expected}, Computed: {computed}",
    )


# Test setup
RTOL = 1e-2
ATOL = 1e-4
FILE = "prob_exeed_d_md.csv"
FUNC = func_probex


@pytest.mark.parametrize("filename", [FILE])
def test_calc_prob_exceed_d_md(load_expected):

    # Inputs
    magnitude = 7
    location = 0.5
    version = "d_md"
    displ = load_expected["displacement"]

    # Expected
    expected = load_expected["probexceed_with_full_aleatory"]

    # Computed
    computed = FUNC(
        magnitude=magnitude,
        location=location,
        displacement_array=displ,
        version=version,
    )

    # Checks
    np.testing.assert_allclose(
        expected,
        computed,
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"For {version} test, Expected: {expected}, Computed: {computed}",
    )
