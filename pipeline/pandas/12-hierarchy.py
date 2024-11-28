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

    df1 = df1.loc[
            (df1['Timestamp'] >= 1417411980) & (df1['Timestamp'] <= 1417417980)
    ]
    df2 = df2.loc[
            (df2['Timestamp'] >= 1417411980) & (df2['Timestamp'] <= 1417417980)
    ]

    df1 = df1.set_index('Timestamp')
    df2 = df2.set_index('Timestamp')

    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])

    df = df.reorder_levels([1, 0], axis=0)

    df = df.sort_index()

    return df
