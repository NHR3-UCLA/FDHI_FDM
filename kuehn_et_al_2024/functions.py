"""Various private helper functions used to calculate the model parameters and predictions."""

# Python imports
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats  # for _calc_transformed_displ

# Model imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kuehn_et_al_2024.load_data import DATA  # for _calc_params

# Model constants
MAG_BREAK, DELTA = 7.0, 0.1


def _func_mode(coefficients, magnitude):
    """
    Calculate magnitude scaling in transformed units.

    Parameters
    ----------
    coefficients : Union[np.recarray, pd.DataFrame]
        A numpy recarray or a pandas DataFrame containing model coefficients.

    magnitude : float
        Earthquake moment magnitude.

    Returns
    -------
    fm : np.array
        Mode in transformed units.
    """
    fm = (
        coefficients["c1"]
        + coefficients["c2"] * (magnitude - MAG_BREAK)
        + (coefficients["c3"] - coefficients["c2"])
        * DELTA
        * np.log(1 + np.exp((magnitude - MAG_BREAK) / DELTA))
    )
    return fm


def _func_mu(coefficients, magnitude, location):
    """
    Calculate mean prediction in transformed units.

    Parameters
    ----------
    coefficients : Union[np.recarray, pd.DataFrame]
        A numpy recarray or a pandas DataFrame containing model coefficients.

    magnitude : float
        Earthquake moment magnitude.

    location : np.array
        Normalized location along rupture length, range [0, 1.0].

    Returns
    -------
    mu : float
        Mean prediction in transformed units.
    """
    fm = _func_mode(coefficients, magnitude=magnitude)

    alpha = coefficients["alpha"]
    beta = coefficients["beta"]
    gamma = coefficients["gamma"]

    a = fm - gamma * np.power(alpha / (alpha + beta), alpha) * np.power(
        beta / (alpha + beta), beta
    )

    mu = a + gamma * np.power(location, alpha) * np.power(1 - location, beta)
    return np.asarray(mu)


def _func_sd_mode_bilinear(coefficients, magnitude):
    """
    Calculate standard deviation of the mode in transformed units.

    Parameters
    ----------
    coefficients : Union[np.recarray, pd.DataFrame]
        A numpy recarray or a pandas DataFrame containing model coefficients.

    magnitude : float
        Earthquake moment magnitude.

    Returns
    -------
    sd: np.array
        Standard deviation of the mode in transformed units.

    Notes
    ------
    Bilinear standard deviation model is only used for strike-slip faulting.
    """
    sd = (
        coefficients["s_m,s1"]
        + coefficients["s_m,s2"] * (magnitude - coefficients["s_m,s3"])
        - coefficients["s_m,s2"]
        * DELTA
        * np.log(1 + np.exp((magnitude - coefficients["s_m,s3"]) / DELTA))
    )
    return np.asarray(sd)


def _func_sd_mode_sigmoid(coefficients, magnitude):
    """
    Calculate standard deviation of the mode in transformed units.

    Parameters
    ----------
    coefficients : Union[np.recarray, pd.DataFrame]
        A numpy recarray or a pandas DataFrame containing model coefficients.

    magnitude : float
        Earthquake moment magnitude.

    Returns
    -------
    sd: np.array
        Standard deviation of the mode in transformed units.

    Notes
    ------
    Sigmoidal standard deviation model is only used for normal faulting.
    """
    sd = coefficients["s_m,n1"] - coefficients["s_m,n2"] / (
        1 + np.exp(-1 * coefficients["s_m,n3"] * (magnitude - MAG_BREAK))
    )
    return np.asarray(sd)


def _func_sd_u(coefficients, location):
    """
    Calculate standard deviation of the location in transformed units.

    Parameters
    ----------
    coefficients : Union[np.recarray, pd.DataFrame]
        A numpy recarray or a pandas DataFrame containing model coefficients.

    location : float
        Normalized location along rupture length, range [0, 1.0].

    Returns
    -------
    sd : np.array
        Standard deviation of the location in transformed units.

    Notes
    ------
    Used only for strike-slip and reverse faulting.
    """
    # Column name2 for stdv coefficients "s_" varies for style of faulting, fix that here
    if isinstance(coefficients, pd.DataFrame):
        s_1 = coefficients["s_s1"] if "s_s1" in coefficients.columns else coefficients["s_r1"]
        s_2 = coefficients["s_s2"] if "s_s2" in coefficients.columns else coefficients["s_r2"]
    elif isinstance(coefficients, np.recarray):
        s_1 = coefficients["s_s1"] if "s_s1" in coefficients.dtype.names else coefficients["s_r1"]
        s_2 = coefficients["s_s2"] if "s_s2" in coefficients.dtype.names else coefficients["s_r2"]
    else:
        raise TypeError(
            "Function argument for model coefficients must be pandas DataFrame or numpy recarray."
        )

    alpha = coefficients["alpha"]
    beta = coefficients["beta"]

    sd = s_1 + s_2 * np.power(location - alpha / (alpha + beta), 2)
    return np.asarray(sd)


def _func_ss(coefficients, magnitude, location):
    """
    Calculate transformation parameter, mean prediction and standard deviations
    (all in transformed units) for strike-slip faulting.

    Parameters
    ----------
    coefficients : Union[np.recarray, pd.DataFrame]
        A numpy recarray or a pandas DataFrame containing model coefficients.

    magnitude : float
        Earthquake moment magnitude.

    location : float
        Normalized location along rupture length, range [0, 1.0].

    Returns
    -------
    Tuple[np.array, np.array, np.array, np.array]
        - 'model_id' : Coefficient row identifier.
        - 'lam' : Box-Cox transformation parameter.
        - 'mu' : Mean prediction in transformed units.
        - 'sd_total' : Total standard deviation in transformed units.
        - 'sd_u' : Within-event standard deviation in transformed units.
        - 'sd_mode' : Between-event standard deviation in transformed units.
    """
    # Calculate mean prediction
    mu = _func_mu(coefficients, magnitude, location)

    # Calculate standard deviations
    sd_mode = _func_sd_mode_bilinear(coefficients, magnitude)
    sd_u = _func_sd_u(coefficients, location)
    sd_total = np.sqrt(np.power(sd_mode, 2) + np.power(sd_u, 2))

    # Transformation parameter
    lam = coefficients["lambda"]

    # Coefficient row identifier
    model_id = coefficients["model_id"]

    return model_id, lam, mu, sd_total, sd_u, sd_mode


def _func_nm(coefficients, magnitude, location):
    """
    Calculate transformation parameter, mean prediction and standard deviations
    (all in transformed units) for normal faulting.

    Parameters
    ----------
    coefficients : Union[np.recarray, pd.DataFrame]
        A numpy recarray or a pandas DataFrame containing model coefficients.

    magnitude : float
        Earthquake moment magnitude.

    location : float
        Normalized location along rupture length, range [0, 1.0].

    Returns
    -------
    Tuple[np.array, np.array, np.array, np.array]
        - 'model_id' : Coefficient row identifier.
        - 'lam' : Box-Cox transformation parameter.
        - 'mu' : Mean prediction in transformed units.
        - 'sd_total' : Total standard deviation in transformed units.
        - 'sd_u' : Within-event standard deviation in transformed units.
        - 'sd_mode' : Between-event standard deviation in transformed units.
    """
    # Calculate mean prediction
    mu = _func_mu(coefficients, magnitude, location)

    # Calculate standard deviations
    sd_mode = _func_sd_mode_sigmoid(coefficients, magnitude)
    sd_u = np.full(len(mu), coefficients["sigma"])
    sd_total = np.sqrt(np.power(sd_mode, 2) + np.power(sd_u, 2))

    # Transformation parameter
    lam = coefficients["lambda"]

    # Coefficient row identifier
    model_id = coefficients["model_id"]

    return model_id, lam, mu, sd_total, sd_u, sd_mode


def _func_rv(coefficients, magnitude, location):
    """
    Calculate transformation parameter, mean prediction and standard deviations
    (all in transformed units) for reverse faulting.

    Parameters
    ----------
    coefficients : Union[np.recarray, pd.DataFrame]
        A numpy recarray or a pandas DataFrame containing model coefficients.

    magnitude : float
        Earthquake moment magnitude.

    location : float
        Normalized location along rupture length, range [0, 1.0].

    Returns
    -------
    Tuple[np.array, np.array, np.array, np.array]
        - 'model_id' : Coefficient row identifier.
        - 'lam' : Box-Cox transformation parameter.
        - 'mu' : Mean prediction in transformed units.
        - 'sd_total' : Total standard deviation in transformed units.
        - 'sd_u' : Within-event standard deviation in transformed units.
        - 'sd_mode' : Between-event standard deviation in transformed units.
    """
    # Calculate mean prediction
    mu = _func_mu(coefficients, magnitude, location)

    # Calculate standard deviations
    sd_mode = np.full(len(mu), coefficients["s_m,r"])
    sd_u = _func_sd_u(coefficients, location)
    sd_total = np.sqrt(np.power(sd_mode, 2) + np.power(sd_u, 2))

    # Transformation parameter
    lam = coefficients["lambda"]

    # Coefficient row identifier
    model_id = coefficients["model_id"]

    return model_id, lam, mu, sd_total, sd_u, sd_mode


def _calc_params(*, magnitude, location, style, coefficient_type="median"):
    """
    Calculate the predicted statistical distribution parameters.

    Parameters
    ----------
    magnitude : int or float
        Earthquake moment magnitude.

    location : int or float
        Normalized location along rupture length, range [0, 1.0].

    style : tr
        Style of faulting (case-insensitive). Valid options are 'strike-slip', 'reverse', or
        'normal'.

    coefficient_type : str, optional
        Option to run model using full epistemic uncertainty or with point estimates (mean or
        median) of the model coefficients (case-insensitive). Valid options are 'mean', 'median',
        or 'full'. Default 'median'.

    Returns
    -------
    tuple
        - 'model_id': Model coefficient row number or point estimate definition.
        - 'bc_param': Box-Cox transformation parameter (lambda).
        - 'mean': Mean displacement in transformed units (unfolded).
        - 'stdv_total': Total standard deviation in transformed units (unfolded).
        - 'stdv_within': Within-event standard deviation in transformed units (unfolded).
        - 'stdv_between': Between-event standard deviation in transformed units (unfolded).
    """
    # Only one value is allowed
    for param, value in zip(
        ["magnitude", "location", "style", "coefficient_type"],
        [magnitude, location, style, coefficient_type],
    ):
        if value is None or isinstance(value, (list, np.ndarray)) and len(value) != 1:
            raise ValueError(f"Only one value is allowed for {param} but {value} was entered.")

    style = style.lower()
    coefficient_type = coefficient_type.lower()
    magnitude = np.atleast_1d(magnitude)
    location = np.atleast_1d(location)

    function_map = {"strike-slip": _func_ss, "reverse": _func_rv, "normal": _func_nm}

    # Calculate parameters for each set of coefficients
    if coefficient_type == "full":
        coeffs = DATA["full"][style].to_records(index=False)
        model_id, lam, mu, sd_total, sd_u, sd_mode = function_map[style](
            coeffs, magnitude, location
        )

    # Calculate parameters for point estimates of coefficients
    else:
        if coefficient_type not in ["mean", "median"]:
            raise ValueError(
                f"'{coefficient_type}' is an invalid 'coefficient_type';"
                " only 'mean', 'median', or 'full' is allowed."
            )

        coeffs = DATA["point"][style]
        coeffs = coeffs[coeffs["model_id"] == coefficient_type].to_records(index=False)

        model_id, lam, mu, sd_total, sd_u, sd_mode = function_map[style](
            coeffs, magnitude, location
        )

    return model_id, lam, mu, sd_total, sd_u, sd_mode


def _calc_analytic_mean(bc_parameter, mean, stdv):
    """
    Helper function to calculate the back-transformed predicted mean displacement in meters
    using the model parameters.

    This analytical solution is from https://robjhyndman.com/hyndsight/backtransforming/

    Parameters
    ----------
    bc_parameter : ArrayLike
        Box-Cox transformation parameter "lambda".

    mean : ArrayLike
        Mean displacement in transformed units.

    stdv : ArrayLike
        Standard deviation of displacement in transformed units.

    Returns
    -------
    float
        Predicted mean displacement in meters.
    """
    return (np.power(bc_parameter * mean + 1, 1 / bc_parameter)) * (
        1 + (np.power(stdv, 2) * (1 - bc_parameter)) / (2 * np.power(bc_parameter * mean + 1, 2))
    )


def _calc_transformed_displ(bc_parameter, mean, stdv, quantile):
    """
    Helper function to calculate predicted displacement in transformed units
    using the model parameters.

    Parameters
    ----------
    bc_parameter : int or float or numpy.ndarray with a single element
        Box-Cox transformation parameter "lambda".

    mean : int or float or numpy.ndarray with a single element
        Mean displacement in transformed units.

    stdv : int or float or numpy.ndarray with a single element
        Standard deviation of displacement in transformed units.

    quantile : int or float or numpy.ndarray with a single element
        Aleatory quantile value. Use -1 for mean.

    Returns
    -------
    displ_bc : int or float or numpy.ndarray with a single element
        Predicted displacement in transformed units.
    """
    if quantile == -1:
        # Compute the back-transformed mean
        displ_meters = _calc_analytic_mean(bc_parameter, mean, stdv)
        displ_bc = (np.power(displ_meters, bc_parameter) - 1) / bc_parameter

    else:
        displ_bc = stats.norm.ppf(quantile, loc=mean, scale=stdv)

    return displ_bc
