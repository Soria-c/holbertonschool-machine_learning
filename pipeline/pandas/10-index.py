#!/usr/bin/env python3
"""Index Timestamp"""


def index(df):
    """
    Sets the 'Timestamp' column as the index of the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing a 'Timestamp' column.

    Returns:
    pd.DataFrame: The DataFrame with the 'Timestamp' column set as the index.
    """
    # Set the 'Timestamp' column as the index
    df_indexed = df.set_index('Timestamp')

    return df_indexed
