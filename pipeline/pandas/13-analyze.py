#!/usr/bin/env python3
"""Descriptive statistics"""

import pandas as pd


def analyze(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes descriptive statistics for all columns except
    the Timestamp column.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: A new DataFrame containing the descriptive statistics for all
                  columns except the Timestamp column.
    """
    # Drop the 'Timestamp' column before computing descriptive statistics
    df_without_timestamp = df.drop(columns=['Timestamp'], errors='ignore')

    # Compute descriptive statistics for the remaining columns
    stats = df_without_timestamp.describe()

    return stats
