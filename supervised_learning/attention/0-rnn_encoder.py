#!/usr/bin/env python3
"""
RNNE Encoder
"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    RNNEncoder class

    Attributes:
        batch (int): The batch size used during training and inference.
        units (int): The number of hidden units in the GRU cell.
        embedding (tf.keras.layers.Embedding): A Keras Embedding layer
        that maps word indices from the vocabulary to embedding vectors.
        gru (tf.keras.layers.GRU): A Keras GRU layer used to encode
        the input sequence.
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Initializes the RNNEncoder.

        Parameters:
            vocab (int): The size of the input vocabulary, which
            determines the number of unique tokens the model can handle.
            embedding (int): The dimensionality of the embedding
            vectors used to represent words in the vocabulary.
            units (int): The number of hidden units in the GRU cell,
            which controls the capacity of the encoder.
            batch (int): The batch size, representing the number
            of samples processed together during training or inference.

        Public attributes:
            self.batch: Stores the batch size.
            self.units: Stores the number of hidden units.
            self.embedding: A Keras Embedding layer that converts word
            indices to dense embedding vectors.
            self.gru: A Keras GRU layer with `units` hidden units,
                      which processes the input sequence.
                      The GRU layer is set to return both the full
                      sequence of outputs and the final hidden state.
                      It uses `glorot_uniform` initialization for
                      the recurrent weights.
        """
        super(RNNEncoder, self).__init__()

        # Public instance attributes
        self.batch = batch
        self.units = units

        # Embedding layer to map words from vocabulary to embedding space
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)

        # GRU layer to encode the sequence
        self.gru = tf.keras.layers.GRU(units=self.units,
                                       # Return the full sequence of outputs
                                       return_sequences=True,
                                       # Return the final hidden state
                                       return_state=True,
                                       # Initialize recurrent weights
                                       recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self):
        """
        Initializes the hidden state for the GRU layer.

        Returns:
            A tensor of shape (batch, units) initialized to zeros, where:
                - batch: The batch size defined in the constructor.
                - units: The number of hidden units in the GRU cell.
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Defines the forward pass of the encoder.

        Parameters:
            x (tf.Tensor): A tensor of shape (batch, input_seq_len) containing
                           the input sequences as word indices.
                           - batch: The batch size.
                           - input_seq_len: The length of each input sequence
                             (number of tokens).
            initial (tf.Tensor): A tensor of shape (batch, units) containing
                                 the initial hidden state for the GRU.
                                 Typically, this is initialized to zeros.

        Returns:
            outputs (tf.Tensor): A tensor of shape (batch, input_seq_len,units)
                                 representing the encoded output
                                 of the GRU for the entire input sequence.
            hidden (tf.Tensor): A tensor of shape (batch, units) representing
                                the final hidden state of the GRU after
                                processing the full input sequence.
        """
        # Convert input word indices into embedding vectors
        x = self.embedding(x)

        # Pass the embeddings into the GRU layer
        outputs, hidden = self.gru(x, initial_state=initial)

        # Return both the full sequence of outputs and the final hidden state
        return outputs, hidden
