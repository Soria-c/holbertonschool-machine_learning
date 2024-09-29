#!/usr/bin/env python3
"""Function to calculate positional encoding for a transformer."""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer model.

    Parameters:
        max_seq_len (int): The maximum sequence length to consider.
        dm (int): The model depth, representing the dimensionality of the
                  embedding vectors used in the transformer.

    Returns:
        np.ndarray: A numpy array of shape (max_seq_len, dm) containing the
                    positional encoding vectors.
    """
    # Initialize the positional encoding matrix
    pos_encoding = np.zeros((max_seq_len, dm))

    # Compute the position indices (0 to max_seq_len-1)
    position = np.arange(max_seq_len)[:, np.newaxis]

    # Compute the depth indices (0 to dm-1)
    div_term = np.exp(np.arange(0, dm, 2) * (-np.log(10000.0) / dm))

    # Apply sine to even indices in the array
    pos_encoding[:, 0::2] = np.sin(position * div_term)

    # Apply cosine to odd indices in the array
    pos_encoding[:, 1::2] = np.cos(position * div_term)

    return pos_encoding
