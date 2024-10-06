#!/usr/bin/env python3
"""
Creates masks for training and validation in a Transformer model.
"""

import tensorflow as tf


def create_masks(inputs, target):
    """
    Creates masks for the encoder and decoder during training.

    Args:
        inputs: A tf.Tensor of shape (batch_size, seq_len_in)
            containing the input sentence.
        target: A tf.Tensor of shape (batch_size, seq_len_out)
            containing the target sentence.

    Returns:
        encoder_mask: Padding mask of shape
            (batch_size, 1, 1, seq_len_in) for the encoder.
        combined_mask: Mask of shape
            (batch_size, 1, seq_len_out, seq_len_out) for the decoder.
        decoder_mask: Padding mask of shape
            (batch_size, 1, 1, seq_len_out) for the decoder.
    """
    # Create the encoder padding mask
    encoder_mask = tf.cast(tf.equal(inputs, 0), tf.float32)
    # Shape: (batch_size, 1, 1, seq_len_in)
    encoder_mask = encoder_mask[tf.newaxis, tf.newaxis, :, :]

    # Create the decoder padding mask
    decoder_padding_mask = tf.cast(tf.equal(target, 0), tf.float32)
    # Shape: (batch_size, 1, 1, seq_len_out)
    decoder_padding_mask = decoder_padding_mask[tf.newaxis, tf.newaxis, :, :]

    # Create lookahead mask for the target (prevent attending to future tokens)
    seq_len_out = tf.shape(target)[1]
    # Upper triangular matrix
    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len_out, seq_len_out)), -1, 0)
    # Shape: (1, 1, seq_len_out, seq_len_out)
    look_ahead_mask = look_ahead_mask[tf.newaxis, tf.newaxis, :, :]

    # Combine the padding and lookahead masks for the combined mask
    # Shape: (batch_size, 1, seq_len_out, seq_len_out)
    combined_mask = tf.maximum(decoder_padding_mask, look_ahead_mask)

    return encoder_mask, combined_mask, decoder_padding_mask
