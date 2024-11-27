#!/usr/bin/env python3
"""
From numpy
"""

import pandas as pd


def from_numpy(array):
    """
    Converts a NumPy ndarray into a pandas DataFrame with
    columns labeled alphabetically (A, B, C, ...).

    Parameters:
    array (np.ndarray): A 2D NumPy array to be converted into a DataFrame.

    Returns:
    pd.DataFrame: A DataFrame created from the input NumPy array with
                  columns labeled A, B, C, ...

    Notes:
    - If the input array is 1D, it will be reshaped into a 2D
      array with one row.
    - The number of columns in the DataFrame will not exceed 26 (A to Z).
    - If the array has more than 26 columns, only the first 26 columns
      will be labeled.
    """
    # Ensure the array is a 2D numpy array
    if len(array.shape) == 1:
        array = array.reshape(1, -1)

    # Create the column labels (A, B, C, ...)
    columns = [chr(65 + i) for i in range(array.shape[1])]

    # Create and return the DataFrame
    return pd.DataFrame(array, columns=columns)
