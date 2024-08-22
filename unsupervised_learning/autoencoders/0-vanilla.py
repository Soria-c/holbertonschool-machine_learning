#!/usr/bin/env python3
"""
Vanilla Autoencoder
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an autoencoder
    """
    # Encoder
    Sequential = keras.Sequential
    Dense = keras.layers.Dense
    Adam = keras.optimizers.Adam

    encoder = Sequential()
    encoder.add(Dense(hidden_layers[0], input_shape=(input_dims,),
                      activation='relu'))

    for units in hidden_layers[1:]:
        encoder.add(Dense(units, activation='relu'))

    encoder.add(Dense(latent_dims, activation='relu'))

    # Decoder
    decoder = Sequential()
    decoder.add(Dense(hidden_layers[-1], input_shape=(latent_dims,),
                      activation='relu'))

    for units in reversed(hidden_layers[:-1]):
        decoder.add(Dense(units, activation='relu'))

    decoder.add(Dense(input_dims, activation='sigmoid'))

    # Autoencoder
    autoencoder = Sequential([encoder, decoder])

    autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy')

    return encoder, decoder, autoencoder
