#!/usr/bin/env python3
"""
Gensim to Keras
"""
import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a trained Gensim Word2Vec model to a Keras Embedding layer.

    Arguments:
    - model: A trained Gensim Word2Vec model.

    Returns:
    - embedding_layer: A trainable Keras Embedding
      layer with the model weights.
    """
    weights = model.wv.vectors
    embedding_layer = tf.keras.layers.Embedding(
            input_dim=weights.shape[0],
            output_dim=weights.shape[1],
            weights=[weights],
            trainable=True
            )
    return embedding_layer
