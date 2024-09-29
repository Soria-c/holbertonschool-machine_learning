#!/usr/bin/env python3
"""Class to perform multi-head attention for transformer models."""

import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    MultiHeadAttention class that applies scaled dot-product attention using
    multiple heads.

    Parameters:
        dm (int): The dimensionality of the model.
        h (int): The number of attention heads.

    Attributes:
        h (int): Number of attention heads.
        dm (int): Dimensionality of the model.
        depth (int): Dimensionality of each attention head (dm // h).
        Wq (Dense): Dense layer to generate query matrices.
        Wk (Dense): Dense layer to generate key matrices.
        Wv (Dense): Dense layer to generate value matrices.
        linear (Dense): Dense layer to project concatenated heads into the
                        output space.
    """

    def __init__(self, dm, h):
        """Initializes the multi-head attention layer."""
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h

        # Dense layers for query, key, and value generation
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)

        # Dense layer for final linear projection
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        Splits the last dimension of the input tensor into (h) heads.

        Parameters:
            x (tf.Tensor): The tensor to split (of shape (batch_size, seq_len,
                          dm)).
            batch_size (int): The batch size.

        Returns:
            tf.Tensor: Reshaped tensor of shape (batch_size, h, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask=None):
        """
        Applies the multi-head attention mechanism.

        Parameters:
            Q (tf.Tensor): Query tensor of shape (batch_size, seq_len_q, dm).
            K (tf.Tensor): Key tensor of shape (batch_size, seq_len_v, dm).
            V (tf.Tensor): Value tensor of shape (batch_size, seq_len_v, dm).
            mask (tf.Tensor, optional): A mask tensor, defaulted to None.

        Returns:
            output (tf.Tensor): Attention output with shape (batch_size,
                                seq_len_q, dm).
            weights (tf.Tensor): Attention weights with shape (batch_size, h,
                                 seq_len_q, seq_len_v).
        """
        batch_size = tf.shape(Q)[0]

        # Linear projections for query, key, and value
        Q = self.Wq(Q)  # (batch_size, seq_len_q, dm)
        K = self.Wk(K)  # (batch_size, seq_len_v, dm)
        V = self.Wv(V)  # (batch_size, seq_len_v, dm)

        # Split Q, K, and V into multiple heads
        Q = self.split_heads(Q, batch_size)  # (batch_size, h, seq_len_q, depth
        K = self.split_heads(K, batch_size)  # (batch_size, h, seq_len_v, depth
        V = self.split_heads(V, batch_size)  # (batch_size, h, seq_len_v, depth

        # Compute scaled dot-product attention for each head
        attention_output, attention_weights = sdp_attention(Q, K, V, mask)

        # Concatenate heads and reshape the output
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output,
                                      (batch_size, -1, self.dm))

        # Apply final linear layer
        output = self.linear(concat_attention)

        return output, attention_weights
