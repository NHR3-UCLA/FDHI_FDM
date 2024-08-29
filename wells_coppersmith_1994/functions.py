"""Various private helper functions used to calculate the model parameters and predictions."""

# Python imports
import numpy as np


def _calc_params_ad(*, magnitude, style="all"):
    """
    Calculate mu and sigma for the AD=f(M) relation in Wells & Coppersmith (1994) Table 2B.

    Parameters
    ----------
    magnitude : int or float
        Earthquake moment magnitude.

    style : str, optional
        Style of faulting (case-insensitive). Default is "all". Valid options are "strike-slip",
        "reverse", "normal", or "all".

    Returns
    -------
    tuple
        - 'mu': Mean prediction for average displacement in log10 units.
        - 'sigma': Standard deviation for average displacement in log10 units.
    """

    coeffs = {
        "all": (-4.8, 0.69, 0.36),
        "strike-slip": (-6.32, 0.90, 0.28),
        "reverse": (-0.74, 0.08, 0.38),
        "normal": (-4.45, 0.63, 0.33),
    }

    magnitude = np.atleast_1d(magnitude)
    style = style.lower()

    a, b, sigma = coeffs[style]
    mu = a + b * magnitude
    sigma = np.full(len(mu), sigma)

    return mu, sigma


def _calc_params_md(*, magnitude, style="all"):
    """
    Calculate mu and sigma for the MD=f(M) relation in Wells & Coppersmith (1994) Table 2B.

    Parameters
    ----------
    magnitude : int or float
        Earthquake moment magnitude.

    style : str, optional
        Style of faulting (case-insensitive). Default is "all". Valid options are "strike-slip",
        "reverse", "normal", or "all".

    Returns
    -------
    tuple
        - 'mu': Mean prediction for maximum displacement in log10 units.
        - 'sigma': Standard deviation for maximum displacement in log10 units.
    """

    coeffs = {
        "all": (-5.46, 0.82, 0.42),
        "strike-slip": (-7.03, 1.03, 0.34),
        "reverse": (-1.84, 0.29, 0.42),
        "normal": (-5.90, 0.89, 0.38),
    }

    magnitude = np.atleast_1d(magnitude)
    style = style.lower()

    a, b, sigma = coeffs[style]
    mu = a + b * magnitude
    sigma = np.full(len(mu), sigma)

    return mu, sigma
