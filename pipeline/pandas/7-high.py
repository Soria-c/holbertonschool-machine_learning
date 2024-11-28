#!/usr/bin/env python3
"""High"""


def high(df):
    """
    Sorts the input DataFrame by the 'High' column in descending order.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing a 'High' column.

    Returns:
    pd.DataFrame: The sorted DataFrame based on the 'High'
    column in descending order.
    """
    # Sort the DataFrame by the 'High' column in descending order
    df_sorted = df.sort_values(by='High', ascending=False)

    return df_sorted
