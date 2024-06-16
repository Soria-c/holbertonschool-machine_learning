#!/usr/bin/env python3
"""Projection Block"""

from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described
    in Deep Residual Learning for Image Recognition (2015)
    """
    F11, F3, F12 = filters

    init = K.initializers.HeNormal(seed=0)

    c1 = K.layers.Conv2D(filters=F11,
                         kernel_size=(1, 1),
                         strides=(s, s),
                         padding="same",
                         kernel_initializer=init)(A_prev)
    n1 = K.layers.BatchNormalization(axis=-1)(c1)
    relu1 = K.layers.Activation(activation="relu")(n1)

    c2 = K.layers.Conv2D(filters=F3,
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         padding="same",
                         kernel_initializer=init)(relu1)
    n2 = K.layers.BatchNormalization(axis=-1)(c2)
    relu2 = K.layers.Activation(activation="relu")(n2)

    c3 = K.layers.Conv2D(filters=F12,
                         kernel_size=(1, 1),
                         strides=(1, 1),
                         padding="same",
                         kernel_initializer=init)(relu2)
    n3 = K.layers.BatchNormalization(axis=-1)(c3)

    c_shortcut = K.layers.Conv2D(filters=F12,
                                 kernel_size=(1, 1),
                                 strides=(s, s),
                                 padding="same",
                                 kernel_initializer=init)(A_prev)
    n_shortcut = K.layers.BatchNormalization(axis=-1)(c_shortcut)

    merged = K.layers.Add()([n3, n_shortcut])

    return K.layers.Activation(activation="relu")(merged)
