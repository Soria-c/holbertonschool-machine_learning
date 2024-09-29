#!/usr/bin/env python3
"""Class to create a decoder block for a transformer."""

import tensorflow as tf

MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
    DecoderBlock class that implements a decoder block for a transformer.

    Parameters:
        dm (int): The dimensionality of the model.
        h (int): The number of attention heads.
        hidden (int): The number of hidden units in the fully connected layer.
        drop_rate (float): The dropout rate. Default is 0.1.

    Attributes:
        mha1 (MultiHeadAttention): First multi-head attention layer.
        mha2 (MultiHeadAttention): Second multi-head attention layer.
        dense_hidden (Dense): Dense layer for hidden output with ReLU
                              activation.
        dense_output (Dense): Dense layer for output with dm units.
        layernorm1 (LayerNormalization): First layer normalization layer.
        layernorm2 (LayerNormalization): Second layer normalization layer.
        layernorm3 (LayerNormalization): Third layer normalization layer.
        dropout1 (Dropout): First dropout layer.
        dropout2 (Dropout): Second dropout layer.
        dropout3 (Dropout): Third dropout layer.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initializes the decoder block layer."""
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Applies the decoder block to the input.

        Parameters:
            x (tf.Tensor): Input tensor of shape (batch, target_seq_len, dm).
            encoder_output (tf.Tensor): Output tensor of the encoder with shape
                                         (batch, input_seq_len, dm).
            training (bool): Boolean to determine if the model is training.
            look_ahead_mask (tf.Tensor): Mask for the first multi-head
                                         attention layer.
            padding_mask (tf.Tensor): Mask for the second multi-head attention
                                      layer.

        Returns:
            tf.Tensor: Output tensor of shape (batch, target_seq_len, dm).
        """
        # First multi-head attention layer
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        x = self.layernorm1(x + self.dropout1(attn1, training=training))

        # Second multi-head attention layer
        attn2, _ = self.mha2(x, encoder_output, encoder_output, padding_mask)
        x = self.layernorm2(x + self.dropout2(attn2, training=training))

        # Feed-forward network
        hidden_output = self.dense_hidden(x)
        output = self.dense_output(hidden_output)

        # Residual connection and layer normalization
        return self.layernorm3(x + self.dropout3(output, training=training))
