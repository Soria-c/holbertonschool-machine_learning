#!/usr/bin/env python3
"""
Vanilla Autoencoder
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates an autoencoder
    """
    Input = keras.Input
    Model = keras.models.Model
    Dense = keras.layers.Dense
    Adam = keras.optimizers.Adam
    l1 = keras.regularizers.L1

    input_layer = Input(shape=(input_dims,))
    encoded = input_layer

    for units in hidden_layers:
        encoded = Dense(units, activation='relu')(encoded)

    latent_layer = Dense(latent_dims, activation='relu',
                         activity_regularizer=l1(lambtha))(encoded)

    # Decoder
    decoded = latent_layer

    for units in reversed(hidden_layers):
        decoded = Dense(units, activation='relu')(decoded)

    encoder = Model(inputs=input_layer, outputs=latent_layer)
    decoder_input = Input(shape=(latent_dims,))
    decoder_output = decoder_input

    for units in reversed(hidden_layers):
        decoder_output = Dense(units, activation='relu')(decoder_output)

    decoder_output = Dense(input_dims, activation='sigmoid')(decoder_output)
    decoder = Model(decoder_input, decoder_output)

    autoencoder = Model(inputs=input_layer,
                        outputs=decoder(encoder(input_layer)))

    autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy')

    return encoder, decoder, autoencoder
