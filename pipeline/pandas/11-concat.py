#!/usr/bin/env python3
"""Concatenate"""


# Assuming the 'index' function from the '10-index' module is already defined
index = __import__('10-index').index


def concat(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenates two DataFrames (`df1` and `df2`) after indexing
    them by the 'Timestamp' column.
    Includes all rows from `df2` up to and including timestamp 1417411920.
    Adds keys to label the rows from `df2` as 'bitstamp' and rows
    from `df1` as 'coinbase'.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame (coinbase).
    df2 (pd.DataFrame): The second DataFrame (bitstamp).

    Returns:
    pd.DataFrame: The concatenated DataFrame with labeled
    rows and indexed by 'Timestamp'.
    """
    # Index both dataframes by the 'Timestamp' column
    df1 = index(df1)
    df2 = index(df2)

    # Filter rows from df2 (bitstamp) where Timestamp is <= 1417411920
    df2_filtered = df2[df2.index <= 1417411920]

    # Concatenate df2 (bitstamp) above df1 (coinbase) with appropriate keys
    concatenated_df = pd.concat([df2_filtered, df1],
                                keys=['bitstamp', 'coinbase'])

    return concatenated_df
