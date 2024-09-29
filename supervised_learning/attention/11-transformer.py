#!/usr/bin/env python3
"""Transformer model for machine translation."""

import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """Transformer model."""

    def __init__(self, N, dm, h, hidden, input_vocab,
                 target_vocab, max_seq_input, max_seq_target,
                 drop_rate=0.1):
        """Class constructor.

        Args:
            N (int): Number of blocks in the encoder and decoder.
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
            hidden (int): Number of hidden units in the fully
                connected layers.
            input_vocab (int): Size of the input vocabulary.
            target_vocab (int): Size of the target vocabulary.
            max_seq_input (int): Maximum sequence length for the input.
            max_seq_target (int): Maximum sequence length for the target.
            drop_rate (float): Dropout rate.

        Attributes:
            encoder (Encoder): Encoder layer of the transformer.
            decoder (Decoder): Decoder layer of the transformer.
            linear (tf.keras.layers.Dense): Final dense layer with
                target_vocab units.
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training,
             encoder_mask, look_ahead_mask, decoder_mask):
        """Forward pass through the transformer.

        Args:
            inputs (tf.Tensor): Input tensor of shape
                (batch, input_seq_len).
            target (tf.Tensor): Target tensor of shape
                (batch, target_seq_len).
            training (bool): Indicates if the model is training.
            encoder_mask (tf.Tensor): Padding mask for the encoder.
            look_ahead_mask (tf.Tensor): Look ahead mask for the
                decoder.
            decoder_mask (tf.Tensor): Padding mask for the decoder.

        Returns:
            tf.Tensor: Output tensor of shape
                (batch, target_seq_len, target_vocab).
        """
        encoder_output = self.encoder(inputs, training, encoder_mask)
        decoder_output = self.decoder(target, encoder_output,
                                      training, look_ahead_mask,
                                      decoder_mask)
        output = self.linear(decoder_output)

        return output
