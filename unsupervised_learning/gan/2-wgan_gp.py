#!/usr/bin/env python3
"""Simple GAN"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_GP(keras.Model):
    """
    WGAN clip class
    """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005, lambda_gp=10):
        """
        Initializes the WGAN_GP model.
        """
        super().__init__()  # run the __init__ of keras.Model first.
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .5  # standard value, but can be changed if necessary
        self.beta_2 = .9  # standard value, but can be changed if necessary

        self.lambda_gp = lambda_gp
        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, delta=1, dtype='int32')
        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

        # define the generator loss and optimizer:
        self.generator.loss = lambda x: -tf.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=self.beta_1,
                beta_2=self.beta_2)
        self.generator.compile(optimizer=generator.optimizer,
                               loss=generator.loss)

        # define the discriminator loss and optimizer:
        self.discriminator.loss = lambda x, y: (
                tf.reduce_mean(y) - tf.reduce_mean(x))
        self.discriminator.optimizer = keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=self.beta_1,
                beta_2=self.beta_2)
        self.discriminator.compile(optimizer=discriminator.optimizer,
                                   loss=discriminator.loss)

    # generator of fake samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        """
        Generates a batch of fake samples using the generator.
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # generator of real samples of size batch_size
    def get_real_sample(self, size=None):
        """
        Generates a batch of real samples.
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    # generator of interpolating samples of size batch_size
    def get_interpolated_sample(self, real_sample, fake_sample):
        """
        Interpolation of samples between real and fake
        """
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape)-u
        return u * real_sample + v * fake_sample

    # computing the gradient penalty
    def gradient_penalty(self, interpolated_sample):
        """
        Grandient penalty for the interpolated samples
        """
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)

    # overloading train_step()
    def train_step(self, useless_argument):
        """
        Train step implementation.
        """
        for _ in range(self.disc_iter):
            # compute the loss for the discriminator
            # in a tape watching the discriminator's weights
            with tf.GradientTape() as tape:
                # get a real sample
                real_samples = self.get_real_sample()
                # get a fake sample
                fake_samples = self.get_fake_sample(training=True)
                # get the interpolated sample
                # (between real and fake computed above)
                interpolated_samples = self.get_interpolated_sample(
                        real_samples, fake_samples)

                real_output = self.discriminator(real_samples, training=True)
                fake_output = self.discriminator(fake_samples, training=True)

                # compute the old loss discr_loss of the discriminator
                # on real and fake samples
                discr_loss = self.discriminator.loss(real_output, fake_output)
                # compute the gradient penalty gp
                gp = self.gradient_penalty(interpolated_samples)
                # compute the sum new_discr_loss =
                # discr_loss + self.lambda_gp * gp
                new_discr_loss = discr_loss + self.lambda_gp * gp

            # apply gradient descent once to the discriminator
            discr_grads = tape.gradient(new_discr_loss,
                                        self.discriminator.trainable_variables)

            self.discriminator.optimizer.apply_gradients(
                    zip(discr_grads,
                        self.discriminator.trainable_variables)
                    )

        # compute the loss for the generator
        # in a tape watching the generator's weights
        with tf.GradientTape() as tape:
            # get a fake sample
            fake_samples = self.get_fake_sample(training=True)
            gen_output = self.discriminator(fake_samples, training=False)
            # compute the loss gen_loss of the generator on this sample
            gen_loss = self.generator.loss(gen_output)

        gen_grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        # apply gradient descent to the discriminator
        self.generator.optimizer.apply_gradients(
                zip(gen_grads,
                    self.generator.trainable_variables)
                )
        # return {"discr_loss": discr_loss, "gen_loss": gen_loss, "gp":gp}
        return {"discr_loss": discr_loss, "gen_loss": gen_loss, "gp": gp}
