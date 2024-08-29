""" """

import pytest
import sys
import numpy as np
from pathlib import Path

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chiou_et_al_2024.calc_avg_displ import func_ad

# Test setup
RTOL = 1e-2
FILE = "avg_displ.csv"
FUNC = func_ad


@pytest.mark.parametrize("filename", [FILE])
def test_avg_displ(load_expected):

    # Inputs
    magnitudes = load_expected["magnitude"]

    # Expected
    expected = load_expected["median_ad"]

    # Computed
    computed = FUNC(magnitude=magnitudes)

    # Checks
    np.testing.assert_allclose(
        expected,
        computed,
        rtol=RTOL,
        err_msg=f"Mag {magnitudes}, Expected: {expected}, Computed: {computed}",
    )
