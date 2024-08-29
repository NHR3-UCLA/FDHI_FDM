"""Various private helper functions used to calculate the model parameters and predictions."""

# Python imports
import numpy as np


def _func_x_star(*, location):
    """
    Calculate the `x*` parameter, which is the elliptical shape scaling parameter.

    Parameters
    ----------
    location : float
        Normalized location along rupture length, range [0, 1.0].

    Returns
    -------
    float
        The elliptical shape scaling parameter.
    """
    x_star = np.sqrt(1 - np.power(1 / 0.5, 2) * np.power(location - 0.5, 2))

    return x_star


def _calc_params_elliptical(*, magnitude, location):
    """
    Calculate the predicted statistical distribution parameters per Eqn 13.

    Parameters
    ----------
    magnitude : Union[float, np.ndarray]
        Earthquake moment magnitude.

    location : Union[float, np.ndarray]
        Normalized location along rupture length, range [0, 1.0].

    Returns
    -------
    tuple
        - 'mu': mean
        - 'sd': standard deviation

    Notes
    ------
    Mu and sd are in natural log units. Exp(mu) is in centimeters, not meters.
    """

    # Convert inputs to list-like numpy arrays
    magnitude, location = map(np.atleast_1d, (magnitude, location))

    a, b, c = 1.7927, 3.3041, -11.2192
    sd = 1.1348

    xstar = _func_x_star(location=location)
    mu = b * xstar + a * magnitude + c
    sd = np.full(len(mu), sd)

    return mu, sd


def _calc_params_quadratic(*, magnitude, location):
    """
    Calculate the predicted statistical distribution parameters per Eqn 10.

    Parameters
    ----------
    magnitude : Union[float, np.ndarray]
        Earthquake moment magnitude.

    location : Union[float, np.ndarray]
        Normalized location along rupture length, range [0, 1.0].

    Returns
    -------
    tuple
        - 'mu': mean
        - 'sd': standard deviation

    Notes
    ------
    Mu and sd are in natural log units. Exp(mu) is in centimeters, not meters.
    """

    # Convert inputs to list-like numpy arrays
    magnitude, location = map(np.atleast_1d, (magnitude, location))

    folded_location = np.minimum(location, 1 - location)

    a, b, c, d = 1.7895, 14.4696, -20.1723, -10.54512
    sd = 1.1346

    mu = a * magnitude + b * folded_location + c * np.power(folded_location, 2) + d
    sd = np.full(len(mu), sd)

    return mu, sd
