#!/usr/bin/env python3
"""Function to calculate the scaled dot product attention."""

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the scaled dot product attention.

    Parameters:
        Q (tf.Tensor): Query matrix with shape (..., seq_len_q, dk).
        K (tf.Tensor): Key matrix with shape (..., seq_len_v, dk).
        V (tf.Tensor): Value matrix with shape (..., seq_len_v, dv).
        mask (tf.Tensor, optional): Mask tensor that can be broadcasted into
                                    shape (..., seq_len_q, seq_len_v). Defaults
                                    to None.

    Returns:
        output (tf.Tensor): A tensor of shape (..., seq_len_q, dv) containing
                            the scaled dot product attention output.
        weights (tf.Tensor): A tensor of shape (..., seq_len_q, seq_len_v)
                             containing the attention weights.
    """
    # Get the dimensionality of the key
    dk = tf.cast(tf.shape(K)[-1], tf.float32)

    # Calculate the scaled dot product of Q and K^T
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # Scale the result by the square root of dk
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Apply the mask if provided
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Calculate the attention weights using softmax
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # Calculate the attention output as a weighted sum of the value matrix V
    output = tf.matmul(attention_weights, V)

    return output, attention_weights
