#!/usr/bin/env python3
"""Flip Switcg"""
import pandas as pd


def flip_switch(df):
    """
    Sorts the input DataFrame in reverse chronological
    order based on the index,
    transposes the sorted DataFrame, and returns
    the transformed DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame, assumed
    to have a datetime-like index.

    Returns:
    pd.DataFrame: The transformed DataFrame sorted
    in reverse chronological order and transposed.
    """
    # Sort the DataFrame in reverse chronological order based on the index
    df_sorted = df.sort_index(ascending=False)

    # Transpose the DataFrame
    df_transposed = df_sorted.T

    return df_transposed
