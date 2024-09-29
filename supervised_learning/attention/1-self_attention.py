#!/usr/bin/env python3
"""
Self Attention
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    SelfAttention is a custom Keras layer that implements the attention
    mechanism
    for machine translation. This layer computes the alignment scores
    between
    the previous decoder hidden state and the encoder hidden states to
    generate a context vector for the decoder.

    Attributes:
        W (tf.keras.layers.Dense): A dense layer applied to the previous
                                   decoder hidden state.
        U (tf.keras.layers.Dense): A dense layer applied to the encoder hidden
                                   states.
        V (tf.keras.layers.Dense): A dense layer applied to the tanh of the sum
                                   of the outputs of W and U.
    """

    def __init__(self, units):
        """
        Initializes the SelfAttention layer.

        Parameters:
            units (int): The number of hidden units in the alignment model,
                         controlling the capacity of the attention mechanism.

        Public attributes:
            self.W: A Dense layer with `units` hidden units, applied to the
                    previous decoder hidden state.
            self.U: A Dense layer with `units` hidden units, applied to the
                    encoder hidden states.
            self.V: A Dense layer with 1 hidden unit, applied to the tanh of
                    the sum of W and U outputs, representing the alignment
                    scores.
        """
        super(SelfAttention, self).__init__()

        # Dense layers for computing attention
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Computes the attention scores and context vector.

        Parameters:
            s_prev (tf.Tensor): A tensor of shape (batch, units) containing the
                                previous decoder hidden state.
            hidden_states (tf.Tensor): A tensor of shape (batch, input_seq_len,
                                       units) containing the encoder's hidden
                                       states.

        Returns:
            context (tf.Tensor): A tensor of shape (batch, units) representing
                                 the context vector for the decoder.
            weights (tf.Tensor): A tensor of shape (batch, input_seq_len, 1)
                                 representing the attention weights over the
                                 encoder hidden states.
        """
        # Add an axis to s_prev to match hidden_states dimensions
        # (batch, 1, units)
        s_prev_expanded = tf.expand_dims(s_prev, 1)

        # Compute alignment scores: W(s_prev) + U(hidden_states)
        score = self.V(tf.nn.tanh(self.W(s_prev_expanded)
                                  + self.U(hidden_states)))

        # Compute attention weights (softmax across the input sequence length)
        weights = tf.nn.softmax(score, axis=1)

        # Compute the context vector as the weighted sum of the hidden states
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        # Return the context vector and attention weights
        return context, weights
