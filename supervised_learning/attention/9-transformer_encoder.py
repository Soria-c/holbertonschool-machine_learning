#!/usr/bin/env python3
"""Class to create the encoder for a transformer."""

import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
    Encoder class that implements an encoder for a transformer.

    Parameters:
        N (int): Number of blocks in the encoder.
        dm (int): Dimensionality of the model.
        h (int): Number of attention heads.
        hidden (int): Number of hidden units in the fully connected layer.
        input_vocab (int): Size of the input vocabulary.
        max_seq_len (int): Maximum sequence length possible.
        drop_rate (float): Dropout rate. Default is 0.1.

    Attributes:
        N (int): Number of blocks in the encoder.
        dm (int): Dimensionality of the model.
        embedding (Embedding): Embedding layer for the inputs.
        positional_encoding (numpy.ndarray): Positional encodings.
        blocks (list): List of EncoderBlock instances.
        dropout (Dropout): Dropout layer for positional encodings.
    """

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """Initializes the encoder layer."""
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Applies the encoder to the input.

        Parameters:
            x (tf.Tensor): Input tensor of shape (batch, input_seq_len).
            training (bool): Boolean to determine if the model is training.
            mask (tf.Tensor): Mask to be applied for multi-head attention.

        Returns:
            tf.Tensor: Output tensor of shape (batch, input_seq_len, dm).
        """
        seq_len = tf.shape(x)[1]
        # Apply embedding
        x = self.embedding(x)
        # Add positional encodings
        x += self.positional_encoding[:seq_len, :]
        x = self.dropout(x, training=training)

        # Pass through each EncoderBlock
        for block in self.blocks:
            x = block(x, training, mask)

        return x
