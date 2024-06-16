#!/usr/bin/env python3
"""Dense Block"""

from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in
    Densely Connected Convolutional Networks:
    """
    init = K.initializers.HeNormal(seed=0)
    concat_features = X

    for _ in range(layers):
        bn1 = K.layers.BatchNormalization(axis=-1)(concat_features)
        relu1 = K.layers.Activation('relu')(bn1)

        c1 = K.layers.Conv2D(4 * growth_rate,
                             (1, 1),
                             padding='same',
                             kernel_initializer=init)(relu1)

        bn2 = K.layers.BatchNormalization(axis=-1)(c1)
        relu2 = K.layers.Activation('relu')(bn2)

        c2 = K.layers.Conv2D(growth_rate,
                             (3, 3), padding='same',
                             kernel_initializer=init)(relu2)

        concat_features = K.layers.Concatenate(axis=-1)([concat_features,
                                                         c2])
        nb_filters += growth_rate
    return concat_features, nb_filters
