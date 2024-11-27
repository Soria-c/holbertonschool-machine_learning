#!/usr/bin/env python3
"""From dictionary"""
import pandas as pd


def create_dataframe():
    """
    Creates a pandas DataFrame with two columns and labeled rows.

    The DataFrame has the following structure:
    - The first column is labeled "First" with the values: 0.0, 0.5, 1.0, 1.5.
    - The second column is labeled "Second" with the values:
            'one', 'two', 'three', 'four'.
    - The rows are labeled A, B, C, and D, respectively.

    Returns:
    pd.DataFrame: A DataFrame created from the specified data.
    """
    # Data dictionary
    data = {
        "First": [0.0, 0.5, 1.0, 1.5],
        "Second": ['one', 'two', 'three', 'four']
    }

    # Row labels
    index = ['A', 'B', 'C', 'D']

    # Create the DataFrame
    df = pd.DataFrame(data, index=index)

    return df


# Create the DataFrame and save it to the variable df
df = create_dataframe()
