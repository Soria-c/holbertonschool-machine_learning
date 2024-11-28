#!/usr/bin/env python3
"""Prune, remove rows"""


def prune(df):
    """
    Removes rows from the DataFrame where the 'Close' column has NaN values.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing a 'Close' column.

    Returns:
    pd.DataFrame: The DataFrame with rows removed where 'Close' has NaN values.
    """
    # Remove rows where 'Close' column has NaN values
    df_cleaned = df.dropna(subset=['Close'])

    return df_cleaned
