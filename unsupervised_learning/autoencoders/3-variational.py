#!/usr/bin/env python3
"""
Variational Autoencoder
"""

# import tensorflow.keras as keras
import keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder
    """
    Dense = keras.layers.Dense
    Input = keras.Input
    Model = keras.models.Model
    Adam = keras.optimizers.Adam
    Lambda = keras.layers.Lambda
    binary_crossentropy = keras.losses.binary_crossentropy
    K = keras.backend

    def sampling(args):
        """
        Reparameterization trick by sampling from an isotropic unit Gaussian.
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    inputs = Input(shape=(input_dims,))
    x = inputs

    for nodes in hidden_layers:
        x = Dense(nodes, activation='relu')(x)

    z_mean = Dense(latent_dims, activation=None)(x)
    z_log_var = Dense(latent_dims, activation=None)(x)

    z = Lambda(sampling, output_shape=(latent_dims,))([z_mean, z_log_var])

    encoder = Model(inputs, [z, z_mean, z_log_var])

    # Decoder
    latent_inputs = Input(shape=(latent_dims,))
    x = latent_inputs

    for nodes in reversed(hidden_layers):
        x = Dense(nodes, activation='relu')(x)

    outputs = Dense(input_dims, activation='sigmoid')(x)

    decoder = Model(latent_inputs, outputs)

    # Full autoencoder
    outputs = decoder(encoder(inputs)[0])
    autoencoder = Model(inputs, outputs)

    # Losses
    reconstruction_loss = binary_crossentropy(inputs, outputs)
    reconstruction_loss *= input_dims

    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = K.mean(reconstruction_loss + kl_loss)
    autoencoder.add_loss(vae_loss)

    # Compile the model
    autoencoder.compile(optimizer=Adam())

    return encoder, decoder, autoencoder
