"""Various private helper functions used to calculate the model parameters and predictions."""

# Python imports
import numpy as np


def _calc_params_ad(*, magnitude):
    """
    Calculate the predicted statistical distribution parameters: mean and standard deviation for
    the AD=f(M) relation in Moss et al. 2022 Table 4.4, "Empirical AD Complete Only."

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

    a, b, sigma = -2.87, 0.416, 0.2
    mu = a + b * np.atleast_1d(magnitude)
    sigma = np.full(len(mu), sigma)
    return mu, sigma


def _calc_params_d_ad(*, location):
    """
    Calculate the predicted statistical distribution parameters: shape (alpha) and scale (beta)
    parameters for Gamma distribution per Figures 4.3 and 4.4 (top eqns) in Moss et al. 2022.

    Parameters
    ----------
    location : int or float
        Normalized location along rupture length, range [0, 1.0].

    Returns
    -------
    tuple
        - 'alpha': Shape parameter for Gamma distribution for D/AD.
        - 'beta': Scale parameter for Gamma distribution for D/AD.
    """

    location = np.atleast_1d(location)
    folded_location = np.minimum(location, 1 - location)
    a1, a2 = 4.2797, 1.6216
    b1, b2 = -0.5003, 0.5133
    alpha = a1 * folded_location + a2
    beta = b1 * folded_location + b2
    return alpha, beta


def _calc_params_md(*, magnitude):
    """
    Calculate the predicted statistical distribution parameters: mean and standard deviation for
    the MD=f(M) relation in Moss et al. 2022 Table 4.4, "Empirical MD Complete Only."

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

    a, b, sigma = -2.5, 0.415, 0.2
    mu = a + b * np.atleast_1d(magnitude)
    sigma = np.full(len(mu), sigma)
    return mu, sigma


def _calc_params_d_md(*, location):
    """
    Calculate the predicted statistical distribution parameters: shape (alpha) and scale (beta)
    parameters for Gamma distribution per Figures 4.3 and 4.4 (bottom eqns) in Moss et al. 2022.

    Parameters
    ----------
    location : int or float
        Normalized location along rupture length, range [0, 1.0].

    Returns
    -------
    tuple
        - 'alpha': Shape parameter for Gamma distribution for D/MD.
        - 'beta': Scale parameter for Gamma distribution for D/MD.
    """

    np.atleast_1d(location)
    folded_location = np.minimum(location, 1 - location)
    a1, a2 = 1.422, 1.856
    b1, b2 = -0.0832, 0.1994
    alpha = a1 * folded_location + a2
    beta = b1 * folded_location + b2

    return alpha, beta
