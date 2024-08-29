""" """

import pytest
import sys
import numpy as np
from pathlib import Path

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kuehn_et_al_2024.calc_prob_exceed import func_probex

# Test setup
RTOL = 1e-2
ATOL = 1e-4
FILE = "prob_exceed_mean_coeffs.csv"
FUNC = func_probex


@pytest.mark.parametrize("filename", [FILE])
def test_calc_prob_exceed_mean_coeffs(load_expected):

    # Inputs
    magnitude = 6.5
    location = 0.25
    style = "normal"
    displ = load_expected["displ_m"]

    # Expected
    expected_folded = load_expected["probex_folded"]
    expected_site = load_expected["probex_site"]

    # Computed
    computed_folded = FUNC(
        magnitude=magnitude,
        location=location,
        style=style,
        displacement_array=displ,
        coefficient_type="mean",
        folded=True,
        debug=False,
    )

    computed_site = FUNC(
        magnitude=magnitude,
        location=location,
        style=style,
        displacement_array=displ,
        coefficient_type="mean",
        folded=False,
        debug=False,
    )

    # Checks
    np.testing.assert_allclose(
        expected_folded,
        computed_folded,
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"For the folded case, Expected: {expected_folded}, Computed: {computed_folded}",
    )

    np.testing.assert_allclose(
        expected_site,
        computed_site,
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"For the unfolded case, Expected: {expected_site}, Computed: {computed_site}",
    )
