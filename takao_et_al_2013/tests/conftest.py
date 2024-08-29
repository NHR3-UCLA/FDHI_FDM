""" """

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def load_expected(filename):
    """Load expected values."""
    filepath = Path(__file__).parent / "expected" / filename
    return np.genfromtxt(filepath, delimiter=",", names=True, encoding="UTF-8-sig", dtype=None)
