"""Various private helper functions used to calculate the model parameters and predictions."""

# Python imports
import numpy as np


def _func_x_star(*, l2L):
    """
    Calculate the `x*` parameter, which is the elliptical shape scaling parameter.

    Parameters
    ----------
    l2L : float
        Normalized location along rupture length, range [0, 1.0].

    Returns
    -------
    float
        The elliptical shape scaling parameter.
    """
    x_star = np.sqrt(1 - np.power(1 / 0.5, 2) * np.power(l2L - 0.5, 2))

    return x_star


def _func_mag(*, magnitude, version="model7"):
    """
    Calculate the magnitude scaling component.

    Parameters
    ----------
    magnitude : float
        The earthquake moment magnitude.

    version : str
        Specify which Chiou et al. nEMG model to use (case-insensitive). The default is `model7`.
        Valid options are `model7`, `model8_1`, `model8_2`, or `model8_3`.
        Only one value is allowed.

    Returns
    -------
    float
        The f_m(M).
    """

    version = version.lower()
    valid_versions = {"model7", "model8_1", "model8_2", "model8_3"}
    if version not in valid_versions:
        raise ValueError(
            f"Invalid version. Please choose one of the valid options: {valid_versions}."
        )

    coeffs_m1 = {"model7": 3.48953, "model8_1": 3.23194, "model8_2": 4.02509, "model8_3": 5.18541}
    coeffs_m2 = {"model7": 0.80407, "model8_1": 0.02095, "model8_2": 1.55817, "model8_3": 1.98653}
    coeffs_m3 = {"model7": 7.10, "model8_1": 7.32, "model8_2": 6.75, "model8_3": 6.40}

    cn = 10
    m1 = coeffs_m1[version]
    m2 = coeffs_m2[version]
    m3 = coeffs_m3[version]

    f_m = m2 * (magnitude - m3) + (m2 - m1) / cn * np.log(
        0.5 * (1 + np.exp(-cn * (magnitude - m3)))
    )

    return f_m


def _func_mu(*, magnitude, l2L, version="model7"):
    """
    Calculate the mean `mu` of the Gaussian component in ln units.

    Parameters
    ----------
    magnitude : float
        The earthquake moment magnitude.

    l2L : float
        Normalized location along rupture length, range [0, 1.0].

    version : str
        Specify which Chiou et al. nEMG model to use (case-insensitive). The default is `model7`.
        Valid options are `model7`, `model8_1`, `model8_2`, or `model8_3`.
        Only one value is allowed.

    Returns
    -------
    float
        The location parameter for a Normal distribution.
    """

    version = version.lower()
    valid_versions = {"model7", "model8_1", "model8_2", "model8_3"}
    if version not in valid_versions:
        raise ValueError(
            f"Invalid version. Please choose one of the valid options: {valid_versions}."
        )

    coeffs_c0 = {"model7": 1.33228, "model8_1": 1.82276, "model8_2": 0.45201, "model8_3": -0.56569}
    coeffs_c1 = {"model7": 2.01791, "model8_1": 2.01712, "model8_2": 2.01730, "model8_3": 2.01927}

    c0 = coeffs_c0[version]
    c1 = coeffs_c1[version]

    x_star = _func_x_star(l2L=l2L)
    f_m = _func_mag(magnitude=magnitude, version=version)

    mu = c0 + f_m + c1 * (x_star - 1)

    return mu


def _func_sd_eq(*, magnitude, version="model7"):
    """
    Calculate the aleatory variability `sigma_eq` on magnitude.

    Parameters
    ----------
    magnitude : float
        The earthquake moment magnitude.

    version : str
        Specify which Chiou et al. nEMG model to use (case-insensitive). The default is `model7`.
        Valid options are `model7`, `model8_1`, `model8_2`, or `model8_3`.
        Only one value is allowed.

    Returns
    -------
    float
        The standard deviation of the event terms.
    """

    version = version.lower()
    valid_versions = {"model7", "model8_1", "model8_2", "model8_3"}
    if version not in valid_versions:
        raise ValueError(
            f"Invalid version. Please choose one of the valid options: {valid_versions}."
        )

    coeffs_cv1 = {"model7": 1.12222, "model8_1": 1.12867, "model8_2": 1.14635, "model8_3": 1.11697}
    coeffs_cv2 = {
        "model7": -0.77666,
        "model8_1": -0.73794,
        "model8_2": -0.74292,
        "model8_3": -0.61806,
    }

    cv1 = coeffs_cv1[version]
    cv2 = coeffs_cv2[version]

    sigma_eq = np.maximum(0.4, cv1 * np.exp(cv2 * np.maximum(0, magnitude - 6.1)))

    return sigma_eq


def _func_sd(*, l2L, version="model7"):
    """
    Calculate the along-strike aleatory variability `sigma`.
    It uses `l2Lf` which is x/L folded to range [0,0.5].
    These coefficients are for Model #7.

    Parameters
    ----------
    l2L : float
        Normalized location along rupture length, range [0, 1.0].

    version : str
        Specify which Chiou et al. nEMG model to use (case-insensitive). The default is `model7`.
        Valid options are `model7`, `model8_1`, `model8_2`, or `model8_3`.
        Only one value is allowed.

    Returns
    -------
    float
        The standard deviation along the rupture.
    """

    version = version.lower()
    valid_versions = {"model7", "model8_1", "model8_2", "model8_3"}
    if version not in valid_versions:
        raise ValueError(
            f"Invalid version. Please choose one of the valid options: {valid_versions}."
        )

    coeffs_cv3 = {"model7": 0.53074, "model8_1": 0.53060, "model8_2": 0.52837, "model8_3": 0.53081}
    coeffs_cv4 = {
        "model7": -3.88902,
        "model8_1": -3.88918,
        "model8_2": -3.90045,
        "model8_3": -3.89073,
    }

    cv3 = coeffs_cv3[version]
    cv4 = coeffs_cv4[version]
    ccap = 0.10877

    l2Lf = np.where(l2L <= 0.5, l2L, 1 - l2L)

    sigma = cv3 * np.exp(cv4 * np.maximum(0, l2Lf - ccap))

    return sigma


def _func_nu(version="model7"):
    """
    Calculate the mean and standard deviation `nu` of the exponential component in ln units.

    Parameters
    ----------
    version : str
        Specify which Chiou et al. nEMG model to use (case-insensitive). The default is `model7`.
        Valid options are `model7`, `model8_1`, `model8_2`, or `model8_3`.
        Only one value is allowed.

    Returns
    -------
    float
        nu.
    """

    version = version.lower()
    valid_versions = {"model7", "model8_1", "model8_2", "model8_3"}
    if version not in valid_versions:
        raise ValueError(
            f"Invalid version. Please choose one of the valid options: {valid_versions}."
        )

    coeffs_cv5 = {"model7": 0.86952, "model8_1": 0.86967, "model8_2": 0.86969, "model8_3": 0.86951}

    cv5 = coeffs_cv5[version]
    nu = cv5

    return nu


def _calc_params(*, magnitude, l2L, version="model7"):
    """
    Calculate the predicted statistical distribution parameters in ln units.

    Parameters
    ----------
    magnitude : int or float
        Earthquake moment magnitude.

    l2L : float
        Normalized location along rupture length, range [0, 1.0].

    version : str
        Specify which Chiou et al. nEMG model to use (case-insensitive). The default is `model7`.
        Valid options are `model7`, `model8_1`, `model8_2`, or `model8_3`.
        Only one value is allowed.

    Returns
    -------
    tuple
        - 'mu': The mean for the Gaussian component.
        - 'nu': The mean and standard deviation of the exponential component
        - 'sd_total': The total aleatory variability for the Gaussian component.
        - 'sd_eq': The standard deviation of the event terms.
        - 'sd': The standard deviation along the rupture.
    """

    version = version.lower()
    valid_versions = {"model7", "model8_1", "model8_2", "model8_3"}
    if version not in valid_versions:
        raise ValueError(
            f"Invalid version. Please choose one of the valid options: {valid_versions}."
        )

    mu = _func_mu(magnitude=magnitude, l2L=l2L, version=version)
    nu = _func_nu(version=version)

    sd_eq = _func_sd_eq(magnitude=magnitude, version=version)
    sd = _func_sd(l2L=l2L, version=version)

    sd_total = np.sqrt(np.power(sd_eq, 2) + np.power(sd, 2))

    return mu, nu, sd_total, sd_eq, sd
