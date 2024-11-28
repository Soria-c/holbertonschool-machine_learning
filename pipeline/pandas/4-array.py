#!/usr/bin/env python3
"""To numpy array"""


def array(df):
    """
    Selects the last 10 rows of the 'High' and 'Close' columns
    from the input DataFrame,
    and converts these values into a numpy.ndarray.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing 'High'
    and 'Close' columns.

    Returns:
    np.ndarray: A numpy array containing the last 10 rows
    of the 'High' and 'Close' columns.
    """
    # Select the last 10 rows of the 'High' and 'Close' columns
    selected_data = df[['High', 'Close']].tail(10)

    # Convert the selected data into a numpy ndarray
    result_array = selected_data.to_numpy()

    return result_array
