#!/usr/bin/env python3
"""Slicing"""


def slice(df):
    """
    Extracts the 'High', 'Low', 'Close', and 'Volume_BTC'
    columns from the input DataFrame,
    selects every 60th row, and returns the resulting sliced DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing
            'High', 'Low', 'Close', and 'Volume_BTC' columns.
    Returns:
    pd.DataFrame: A DataFrame containing every 60th row
        from the selected columns.
    """
    # Extract the specified columns
    selected_columns = df[['High', 'Low', 'Close', 'Volume_(BTC)']]

    # Select every 60th row using slicing
    sliced_df = selected_columns.iloc[::60, :]

    return sliced_df
