#!/usr/bin/env python3
"""Inception Network"""
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    builds the inception network
    as described in Going Deeper with Convolutions (2014)
    """
    input_layer = K.Input(shape=(224, 224, 3))
    c1 = K.layers.Conv2D(filters=64,
                         padding='same',
                         activation='relu',
                         kernel_size=(7, 7),
                         strides=(2, 2),)(input_layer)
    p1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                               strides=(2, 2),
                               padding='same')(c1)
    c2 = K.layers.Conv2D(filters=64,
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         padding='same',
                         activation='relu')(p1)
    c3 = K.layers.Conv2D(filters=192,
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         padding='same',
                         activation='relu')(c2)
    p2 = K.layers.MaxPooling2D(pool_size=(3, 3),
                               strides=(2, 2),
                               padding='same')(c3)
    inc1 = inception_block(p2, (64, 96, 128, 16, 32, 32))
    inc2 = inception_block(inc1, (128, 128, 192, 32, 96, 64))
    p3 = K.layers.MaxPooling2D(pool_size=(3, 3),
                               strides=(2, 2),
                               padding='same')(inc2)
    inc3 = inception_block(p3, (192, 96, 208, 16, 48, 64))
    inc4 = inception_block(inc3, (160, 112, 224, 24, 64, 64))
    inc5 = inception_block(inc4, (128, 128, 256, 24, 64, 64))
    inc6 = inception_block(inc5, (112, 144, 288, 32, 64, 64))
    inc7 = inception_block(inc6, (256, 160, 320, 32, 128, 128))
    p4 = K.layers.MaxPooling2D(pool_size=(3, 3),
                               strides=(2, 2),
                               padding='same')(inc7)
    inc8 = inception_block(p4, (256, 160, 320, 32, 128, 128))
    inc9 = inception_block(inc8, (384, 192, 384, 48, 128, 128))
    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                         strides=(1, 1),
                                         padding='valid')(inc9)
    dropout = K.layers.Dropout(rate=(0.4))(avg_pool)
    fc = K.layers.Dense(units=(1000), activation='softmax',
                        )(dropout)
    return K.Model(inputs=input_layer, outputs=fc)
