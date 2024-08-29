""" """

import pytest
import sys
import numpy as np
from pathlib import Path

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from takao_et_al_2013.calc_deterministic import func_det

# Test setup
RTOL = 2e-2
FILE = "deterministic.csv"
FUNC = func_det


@pytest.mark.parametrize("filename", [FILE])
def test_calc_deterministic(load_expected):

    for row in load_expected:
        # Inputs
        version, srl, magnitude, location, percentile, expected = row

        # Computed
        computed = func_det(
            magnitude=magnitude,
            location=location,
            percentile=percentile,
            srl_km=srl,
            version=version,
        )

        # Checks
        np.testing.assert_allclose(
            expected,
            computed,
            rtol=RTOL,
            err_msg=(
                f"Mag {magnitude}, loc {location}, SRL {srl}, percentile {percentile}, "
                f"Expected: {expected}, Computed: {computed}"
            ),
        )
