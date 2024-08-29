""" """

import pytest
import sys
import numpy as np
from pathlib import Path

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from moss_et_al_2024.calc_deterministic import func_det

# Test setup
RTOL = 2e-2
FILE = "deterministic.csv"
FUNC = func_det


@pytest.mark.parametrize("filename", [FILE])
def test_calc_deterministic(load_expected):

    for row in load_expected:
        # Inputs
        version, magnitude, location, percentile, expected = row

        # Computed
        computed = func_det(
            magnitude=magnitude,
            location=location,
            percentile=percentile,
            version=version,
        )

        # Checks
        np.testing.assert_allclose(
            expected,
            computed,
            rtol=RTOL,
            err_msg=(
                f"Mag {magnitude}, loc {location}, percentile {percentile}, "
                f"Expected: {expected}, Computed: {computed}"
            ),
        )
