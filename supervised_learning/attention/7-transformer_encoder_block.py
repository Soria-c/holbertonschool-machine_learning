#!/usr/bin/env python3
"""Class to create an encoder block for a transformer."""

import tensorflow as tf

MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    EncoderBlock class that implements an encoder block for a transformer.

    Parameters:
        dm (int): The dimensionality of the model.
        h (int): The number of attention heads.
        hidden (int): The number of hidden units in the fully connected layer.
        drop_rate (float): The dropout rate. Default is 0.1.

    Attributes:
        mha (MultiHeadAttention): Multi-head attention layer.
        dense_hidden (Dense): Dense layer for the hidden output with ReLU
                              activation.
        dense_output (Dense): Dense layer for the output with dm units.
        layernorm1 (LayerNormalization): First layer normalization layer.
        layernorm2 (LayerNormalization): Second layer normalization layer.
        dropout1 (Dropout): First dropout layer.
        dropout2 (Dropout): Second dropout layer.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initializes the encoder block layer."""
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        Applies the encoder block to the input.

        Parameters:
            x (tf.Tensor): Input tensor of shape (batch, input_seq_len, dm).
            training (bool): Boolean to determine if the model is training.
            mask (tf.Tensor, optional): Mask to apply for multi-head attention.

        Returns:
            tf.Tensor: Output tensor of shape (batch, input_seq_len, dm).
        """
        # Multi-head attention
        attn_output, _ = self.mha(x, x, x, mask)

        # Residual connection and layer normalization
        x = self.layernorm1(x + self.dropout1(attn_output, training=training))

        # Feed-forward network
        hidden_output = self.dense_hidden(x)
        output = self.dense_output(hidden_output)

        # Residual connection and layer normalization
        return self.layernorm2(x + self.dropout2(output, training=training))
