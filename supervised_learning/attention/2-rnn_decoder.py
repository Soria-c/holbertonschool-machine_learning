#!/usr/bin/env python3
"""RNNDecoder class to decode sequences for machine translation."""

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    RNNDecoder is a custom Keras layer designed to decode input sequences for
    machine translation. It takes the previous target sequence word, the
    previous decoder hidden state, and the encoder hidden states to generate
    the next word in the target sequence.

    Attributes:
        embedding (tf.keras.layers.Embedding): A Keras Embedding layer that
                                               maps word indices from the
                                               target vocabulary to embedding
                                               vectors.
        gru (tf.keras.layers.GRU): A GRU layer with a specified number of units
                                   that processes the input sequence and
                                   returns both the full output and the final
                                   hidden state.
        F (tf.keras.layers.Dense): A Dense layer used to generate the final
                                   output as a one-hot vector over the target
                                   vocabulary.
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Initializes the RNNDecoder.

        Parameters:
            vocab (int): Size of the output vocabulary, which determines the
                         number of unique tokens the decoder can output.
            embedding (int): Dimensionality of the embedding vectors used to
                             represent words in the target vocabulary.
            units (int): The number of hidden units in the GRU cell.
            batch (int): The batch size, representing the number of samples
                         processed together during training or inference.

        Public attributes:
            self.embedding: A Keras Embedding layer that converts word indices
                            to dense embedding vectors.
            self.gru: A Keras GRU layer with `units` hidden units, returning
                      both the full sequence of outputs and the final hidden
                      state.
            self.F: A Dense layer with `vocab` units that maps the GRU output
                    to the target vocabulary's one-hot representation.
        """
        super(RNNDecoder, self).__init__()

        # Embedding layer for target vocabulary
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        # GRU layer for sequence decoding
        self.gru = tf.keras.layers.GRU(units=units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        # Dense layer to produce the output word
        self.F = tf.keras.layers.Dense(units=vocab)

        # Self-attention mechanism for alignment
        self.attention = SelfAttention(units=units)

    def call(self, x, s_prev, hidden_states):
        """
        Performs the forward pass of the decoder.

        Parameters:
            x (tf.Tensor): A tensor of shape (batch, 1) containing the previous
                           word in the target sequence as an index of the
                           target vocabulary.
            s_prev (tf.Tensor): A tensor of shape (batch, units) containing the
                                previous decoder hidden state.
            hidden_states (tf.Tensor): A tensor of shape (batch, input_seq_len,
                                       units) containing the encoder's hidden
                                       states.

        Returns:
            y (tf.Tensor): A tensor of shape (batch, vocab) containing the
                           output word as a one-hot vector in the target
                           vocabulary.
            s (tf.Tensor): A tensor of shape (batch, units) containing the new
                           decoder hidden state.
        """
        # Compute attention context using the encoder hidden states
        context, _ = self.attention(s_prev, hidden_states)

        # Convert input word (x) to embedding
        x = self.embedding(x)

        # Concatenate the context vector and the input
        # embedding
        # (in this order)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)

        # Pass the concatenated vector through the GRU
        output, s = self.gru(x)

        output = tf.squeeze(output, axis=1)  # Remove the time dimension

        # Generate output word as a one-hot vector in the target vocabulary
        y = self.F(output)

        # Return output word and updated hidden state

        return y, s
