""" """

import pytest
import sys
import numpy as np
from pathlib import Path

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from petersen_et_al_2011.calc_prob_exceed import func_probex

# Test setup
RTOL = 1e-2
ATOL = 1e-4
FILE = "prob_exceed_elliptical.csv"
FUNC = func_probex


@pytest.mark.parametrize("filename", [FILE])
def test_calc_prob_exceed_d_ad(load_expected):

    # Inputs
    magnitude = 7
    location = 0.5
    version = "elliptical"
    displ = load_expected["displacement"]

    # Expected
    expected = load_expected["probexceed"]

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
