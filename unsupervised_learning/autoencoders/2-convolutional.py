#!/usr/bin/env python3
"""
Convolutional Autoencoder
"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    creates a convolutional autoencoder
    """
    Input = keras.Input
    Model = keras.models.Model
    Adam = keras.optimizers.Adam
    Conv2D = keras.layers.Conv2D
    MaxPooling2D = keras.layers.MaxPool2D
    UpSampling2D = keras.layers.UpSampling2D

    input_layer = Input(shape=input_dims)
    output_layer = Input(shape=latent_dims)

    encoded_layer = input_layer
    decoded_layer = output_layer

    for filter in filters:
        encoded_layer = Conv2D(
            filter, (3, 3), padding="same", activation="relu")(encoded_layer)
        encoded_layer = MaxPooling2D((2, 2), padding="same")(encoded_layer)

    for x, filter in enumerate(reversed(filters)):
        if x == len(filters) - 1:
            decoded_layer = Conv2D(
                filter, (3, 3), padding="valid",
                activation="relu")(decoded_layer)
        else:
            decoded_layer = Conv2D(
                filter, (3, 3), padding="same",
                activation="relu")(decoded_layer)

        decoded_layer = UpSampling2D((2, 2))(decoded_layer)

    decoded_layer = Conv2D(
        input_dims[-1], (3, 3), padding="same", activation="sigmoid"
        )(decoded_layer)

    encoder = Model(input_layer, encoded_layer)

    decoder = Model(output_layer, decoded_layer)

    full_autoencoder = Model(input_layer, decoder(encoder(input_layer)))

    full_autoencoder.compile(optimizer=Adam(), loss="binary_crossentropy")

    return encoder, decoder, full_autoencoder
