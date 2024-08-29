"""This module loads the model coefficients."""

# Python imports
from pathlib import Path
from typing import Union

import os
import pandas as pd

# Define paths and files
dir_data = Path(os.path.join(os.path.dirname(__file__), "data"))

full_posterior_files = {
    "strike-slip": "coefficients_posterior_SS_powtr.csv",
    "reverse": "coefficients_posterior_REV_powtr.csv",
    "normal": "coefficients_posterior_NM_powtr.csv",
}

point_posterior_files = {
    "strike-slip": "coefficients_mean_SS_powtr.csv",
    "reverse": "coefficients_mean_REV_powtr.csv",
    "normal": "coefficients_mean_NM_powtr.csv",
}


# Function to load data
def _load_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load model coefficients.

    Parameters
    ----------
    filepath : Union[str, pathlib.Path]
        The path to the CSV file containing the model coefficients.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the model coefficients.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the provided filepath.
    """
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    try:
        data = pd.read_csv(filepath).rename(columns={"Unnamed: 0": "model_id"})
    except FileNotFoundError:
        raise FileNotFoundError(f"File {filepath.resolve()} not found.")

    return data


# Import full model coefficients
full_posterior_ss = _load_data(dir_data / full_posterior_files["strike-slip"])
full_posterior_rv = _load_data(dir_data / full_posterior_files["reverse"])
full_posterior_nm = _load_data(dir_data / full_posterior_files["normal"])

# Import point model coefficients
point_posterior_ss = _load_data(dir_data / point_posterior_files["strike-slip"])
point_posterior_rv = _load_data(dir_data / point_posterior_files["reverse"])
point_posterior_nm = _load_data(dir_data / point_posterior_files["normal"])

# Create data dictionary
DATA = {
    "full": {
        "strike-slip": full_posterior_ss,
        "reverse": full_posterior_rv,
        "normal": full_posterior_nm,
    },
    "point": {
        "strike-slip": point_posterior_ss,
        "reverse": point_posterior_rv,
        "normal": point_posterior_nm,
    },
}
