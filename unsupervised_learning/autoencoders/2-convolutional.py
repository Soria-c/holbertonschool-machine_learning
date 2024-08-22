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
    x = input_layer

    # Encoder
    for f in filters:
        x = Conv2D(f, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

    # Latent space representation
    latent = Conv2D(filters[-1], (3, 3), activation='relu', padding='same')(x)

    # Decoder
    x = latent

    for f in reversed(filters[:-1]):
        x = Conv2D(f, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)

    # The second to last convolution with 'valid' padding
    x = Conv2D(filters[0], (3, 3), activation='relu', padding='valid')(x)

    # Adding one more upsampling layer to match the original dimensions
    x = UpSampling2D((2, 2))(x)

    # Final convolution to match the input dimensions
    decoder_output = Conv2D(input_dims[-1], (3, 3), activation='sigmoid',
                            padding='same')(x)

    # Encoder model
    encoder = Model(inputs=input_layer, outputs=latent)

    # Decoder model
    decoder_input = Input(shape=latent_dims)
    x = decoder_input

    decoder = Model(decoder_input, decoder_output)

    # Full autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=decoder_output)

    # Compile the autoencoder model
    autoencoder.compile(optimizer="Adam()", loss='binary_crossentropy')

    return encoder, decoder, autoencoder
