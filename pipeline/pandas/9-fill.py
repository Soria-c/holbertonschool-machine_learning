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
    # Remove the 'Weighted_Price' column
    df_cleaned = df.drop(columns=['Weighted_Price'], errors='ignore')

    # Fill missing values in 'Close' with the previous row's value
    df_cleaned['Close'] = df_cleaned['Close'].fillna(method='ffill')

    # Fill missing values in 'High', 'Low', and 'Open' with the corresponding
    # 'Close' value
    df_cleaned[['High', 'Low', 'Open']] = df_cleaned[['High', 'Low', 'Open']]\
        .fillna(df_cleaned['Close'])

    # Set missing values in 'Volume_(BTC)' and 'Volume_(Currency)' to 0
    df_cleaned[['Volume_(BTC)', 'Volume_(Currency)']] = df_cleaned[
        ['Volume_(BTC)', 'Volume_(Currency)']].fillna(0)

    return df_cleaned
