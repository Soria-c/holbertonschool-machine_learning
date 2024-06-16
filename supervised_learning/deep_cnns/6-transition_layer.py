#!/usr/bin/env python3
"""Transition Layer"""

from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in
    Densely Connected Convolutional Networks
    """
    init = K.initializers.HeNormal(seed=0)
    compressed_filters = int(nb_filters * compression)
    bn = K.layers.BatchNormalization(axis=-1)(X)
    relu = K.layers.Activation('relu')(bn)
    conv = K.layers.Conv2D(compressed_filters,
                           (1, 1), padding='same',
                           kernel_initializer=init)(relu)
    avg_pool = K.layers.AveragePooling2D((2, 2),
                                         strides=(2, 2),
                                         padding='same')(conv)

    return avg_pool, compressed_filters
