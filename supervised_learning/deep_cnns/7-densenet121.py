#!/usr/bin/env python3
"""DenseNet-121 Implementation"""

from tensorflow import keras as K

dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture as described
    in Densely Connected Convolutional Networks
    """
    init = K.initializers.HeNormal(seed=0)
    input_shape = (224, 224, 3)

    input_x = K.Input(shape=input_shape)

    X = K.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same',
                        kernel_initializer=init)(input_x)
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.ReLU()(X)
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)

    X, nb_filters = dense_block(X, 64, growth_rate, 6)

    X, nb_filters = transition_layer(X, nb_filters, compression)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)

    X, nb_filters = transition_layer(X, nb_filters, compression)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)

    X, nb_filters = transition_layer(X, nb_filters, compression)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.ReLU()(X)

    X = K.layers.GlobalAveragePooling2D()(X)

    X = K.layers.Dense(1000, activation='softmax', kernel_initializer=init)(X)
    return K.Model(inputs=input_x, outputs=X)
