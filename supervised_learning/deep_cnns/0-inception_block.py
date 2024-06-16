#!/usr/bin/env python3
"""Inception Block"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block as described
    in Going Deeper with Convolutions (2014)
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    init = K.initializers.HeNormal()
    c1 = K.layers.Conv2D(filters=F1,
                         kernel_size=(1, 1),
                         activation='relu',
                         padding='same',
                         kernel_initializer=init)(A_prev)
    c2 = K.layers.Conv2D(filters=F3R,
                         kernel_size=(1, 1),
                         activation='relu',
                         padding='same',
                         kernel_initializer=init)(A_prev)
    c3 = K.layers.Conv2D(filters=F3,
                         kernel_size=(3, 3),
                         activation='relu',
                         padding='same',
                         kernel_initializer=init)(c2)
    c4 = K.layers.Conv2D(filters=F5R,
                         kernel_size=(1, 1),
                         activation='relu',
                         padding='same',
                         kernel_initializer=init)(A_prev)
    c5 = K.layers.Conv2D(filters=F5,
                         kernel_size=(5, 5),
                         activation='relu',
                         padding='same',
                         kernel_initializer=init)(c4)
    p1 = K.layers.MaxPooling2D(pool_size=(3, 3), padding='same',
                               strides=(1, 1))(A_prev)
    c6 = K.layers.Conv2D(filters=FPP,
                         kernel_size=(1, 1),
                         activation='relu',
                         padding='same',
                         kernel_initializer=init)(p1)
    return K.layers.concatenate(inputs=[c1, c3, c5, c6])
