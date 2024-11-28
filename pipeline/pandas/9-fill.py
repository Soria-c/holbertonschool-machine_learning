#!/usr/bin/env python3
"""Fill missing value"""


def fill(df):
    """
    Performs the following operations on the input DataFrame:
    1. Removes the 'Weighted_Price' column.
    2. Fills missing values in the 'Close' column with the previous row's
       value.
    3. Fills missing values in the 'High', 'Low', and 'Open' columns
       with the corresponding 'Close' value.
    4. Sets missing values in the 'Volume_(BTC)' and
       'Volume_(Currency)' columns to 0.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be modified.

    Returns:
    pd.DataFrame: The modified DataFrame after filling missing
    values and removing the 'Weighted_Price' column.
    """
    df = df.drop(columns=['Weighted_Price'])
    # Missing values in Close should be set to previous row value
    df['Close'].fillna(method='pad', inplace=True)
    # Missing values in High, Low, Open should be set to same row's Close value
    df['High'].fillna(df.Close, inplace=True)
    df['Low'].fillna(df.Close, inplace=True)
    df['Open'].fillna(df.Close, inplace=True)
    # Missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
    df['Volume_(BTC)'].fillna(value=0, inplace=True)
    df['Volume_(Currency)'].fillna(value=0, inplace=True)

    return df
