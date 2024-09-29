#!/usr/bin/env python3
"""Decoder class for transformer architecture."""

import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """Decoder for the transformer model."""

    def __init__(self, N, dm, h, hidden, target_vocab,
                 max_seq_len, drop_rate=0.1):
        """Class constructor.

        Args:
            N (int): Number of blocks in the decoder.
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
            hidden (int): Number of hidden units in the fully
                connected layer.
            target_vocab (int): Size of the target vocabulary.
            max_seq_len (int): Maximum sequence length possible.
            drop_rate (float): Dropout rate.

        Attributes:
            N (int): Number of blocks in the decoder.
            dm (int): Dimensionality of the model.
            embedding (tf.keras.layers.Embedding): Embedding layer for
                targets.
            positional_encoding (numpy.ndarray): Positional encodings of
                shape (max_seq_len, dm).
            blocks (list): List containing DecoderBlock instances.
            dropout (tf.keras.layers.Dropout): Dropout layer for
                positional encodings.
        """
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Forward pass through the decoder.

        Args:
            x (tf.Tensor): Input tensor of shape
                (batch, target_seq_len, dm).
            encoder_output (tf.Tensor): Output tensor from the
                encoder of shape (batch, input_seq_len, dm).
            training (bool): Indicates if the model is training.
            look_ahead_mask (tf.Tensor): Mask for the first
                multi-head attention layer.
            padding_mask (tf.Tensor): Mask for the second
                multi-head attention layer.

        Returns:
            tf.Tensor: Output tensor of shape
            (batch, target_seq_len, dm).
        """
        seq_len = tf.shape(x)[1]
        # Create positional encodings
        x += self.positional_encoding[:seq_len, :]
        x = self.dropout(x, training=training)

        # Pass through decoder blocks
        for block in self.blocks:
            x = block(x, encoder_output, training,
                      look_ahead_mask, padding_mask)

        return x
