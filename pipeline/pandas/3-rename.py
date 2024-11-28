#!/usr/bin/env python3
"""Rename"""
import pandas as pd


def rename(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames the 'Timestamp' column to 'Datetime',
    converts the values to datetime,
    and returns a DataFrame containing only the
    'Datetime' and 'Close' columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing a 'Timestamp' column.

    Returns:
    pd.DataFrame: The modified DataFrame with 'Timestamp'
    renamed to 'Datetime' and converted to datetime.
    """
    # Rename 'Timestamp' column to 'Datetime'
    df = df.rename(columns={'Timestamp': 'Datetime'})

    # Convert 'Datetime' column to datetime type
    df['Datetime'] = pd.to_datetime(df['Datetime'])

    # Select only the 'Datetime' and 'Close' columns
    df = df[['Datetime', 'Close']]

    return df
