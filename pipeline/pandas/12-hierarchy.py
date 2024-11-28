#!/usr/bin/env python3
"""Hierarchy"""
import pandas as pd

# Assuming the 'index' function from the '10-index' module is already defined
index = __import__('10-index').index


def hierarchy(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenates two DataFrames (`df1` and `df2`), with Timestamp as the
    first level of the MultiIndex.
    Filters the data between timestamps 1417411980 and 1417417980 (inclusive).
    Labels rows from `df2` (bitstamp) and `df1` (coinbase), ensuring
    the data is in chronological order.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame (coinbase).
    df2 (pd.DataFrame): The second DataFrame (bitstamp).

    Returns:
    pd.DataFrame: The concatenated DataFrame with Timestamp as the first
                  level of the index,
                  ordered chronologically, and with appropriate keys.
    """
    # Index both dataframes by the 'Timestamp' column
    df1 = index(df1)
    df2 = index(df2)

    # Filter rows from both DataFrames for the timestamp range
    # between 1417411980 and 1417417980
    df1_filtered = df1[(df1.index >= 1417411980) & (df1.index <= 1417417980)]
    df2_filtered = df2[(df2.index >= 1417411980) & (df2.index <= 1417417980)]

    # Concatenate df2 (bitstamp) above df1 (coinbase) with appropriate keys
    concatenated_df = pd.concat([df2_filtered, df1_filtered],
                                keys=['bitstamp', 'coinbase'])

    # Ensure the data is sorted by the Timestamp
    # (which is now part of the index)
    concatenated_df = concatenated_df.sort_index(level=0)

    return concatenated_df
