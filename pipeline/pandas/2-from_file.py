#!/usr/bin/env python3
"""From file"""
import pandas as pd


def from_file(filename: str, delimiter: str) -> pd.DataFrame:
    """
    Loads data from a file and returns it as a pandas DataFrame.

    Parameters:
    filename (str): The file path to load the data from.
    delimiter (str): The column separator used in the file.

    Returns:
    pd.DataFrame: The DataFrame containing the data loaded from the file.
    """
    # Load the data from the file using the provided delimiter
    df = pd.read_csv(filename, delimiter=delimiter)

    return df
