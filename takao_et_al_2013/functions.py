"""Various private helper functions used to calculate the model parameters and predictions."""

# Python imports
import numpy as np


def _calc_params_ad(*, magnitude):
    """
    Calculate mu and sigma for the AD=f(M) relation in Takao et al (2013).

    Parameters
    ----------
    magnitude : int or float
        Earthquake moment magnitude.

    Returns
    -------
    tuple
        - 'mu': Mean prediction for average displacement in log10 units.
        - 'sigma': Standard deviation for average displacement in log10 units.
    """

    a, b, sigma = -4.80, 0.69, 0.36
    mu = a + b * np.atleast_1d(magnitude)
    sigma = np.full(len(mu), sigma)
    return mu, sigma


def _calc_params_d_ad(*, location, srl_km):
    """
    Calculate alpha and beta based on location and surface rupture length.

    Parameters
    ----------
    location : int or float
        Normalized location along rupture length, range [0, 1.0].

    srl_km : int or float
        Total surface rupture length in kilometers.

    Returns
    -------
    tuple
        - 'alpha': Shape parameter for Gamma distribution for D/AD.
        - 'beta': Scale parameter for Gamma distribution for D/AD.
    """

    if srl_km < 10:
        alpha, beta = 1.53, 0.58
    else:
        np.atleast_1d(location)
        folded_location = np.minimum(location, 1 - location)
        alpha = np.exp(0.70 + 0.34 * folded_location)
        beta = np.exp(-1.40 + 1.82 * folded_location)
    return alpha, beta


def _calc_params_md(*, magnitude):
    """
    Calculate mu and sigma for the MD=f(M) relation in Takao et al (2013).

    Parameters
    ----------
    magnitude : int or float
        Earthquake moment magnitude.

    Returns
    -------
    tuple
        - 'mu': Mean prediction for maximum displacement in log10 units.
        - 'sigma': Standard deviation for maximum displacement in log10 units.
    """

    a, b, sigma = -5.16, 0.82, 0.42
    mu = a + b * np.atleast_1d(magnitude)
    sigma = np.full(len(mu), sigma)
    return mu, sigma


def _calc_params_d_md(*, location, srl_km):
    """
    Calculate alpha and beta based on location and surface rupture length.

    Parameters
    ----------
    location : int or float
        Normalized location along rupture length, range [0, 1.0].

    srl_km : int or float
        Total surface rupture length in kilometers.

    Returns
    -------
    tuple
        - 'alpha': Shape parameter for Beta distribution for D/MD.
        - 'beta': Shape parameter for Beta distribution for D/MD.
    """

    if srl_km < 10:
        alpha, beta = 0.91, 1.9
    else:
        np.atleast_1d(location)
        folded_location = np.minimum(location, 1 - location)
        alpha = np.exp(0.70 - 0.87 * folded_location)
        beta = np.exp(2.30 - 3.84 * folded_location)
    return alpha, beta
