#!/usr/bin/env python3
"""LeNet-5 (Tensorflow 1)"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
    Builds a modified version of the LeNet-5 architecture using tensorflow
    """
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
    conv1 = tf.layers.Conv2D(filters=6,
                             kernel_size=(5, 5),
                             padding='same',
                             activation='relu',
                             kernel_initializer=initializer)(x)
    pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2))(conv1)
    conv2 = tf.layers.Conv2D(filters=16,
                             kernel_size=(5, 5),
                             padding='valid',
                             activation='relu',
                             kernel_initializer=initializer)(pool1)
    pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2))(conv2)
    flat = tf.layers.Flatten()(pool2)
    fc1 = tf.layers.Dense(units=120,
                          kernel_initializer=initializer,
                          activation='relu')(flat)
    fc2 = tf.layers.Dense(units=84,
                          kernel_initializer=initializer,
                          activation='relu')(fc1)
    fc3 = tf.layers.Dense(units=10,
                          kernel_initializer=initializer)(fc2)
    softmax_output = tf.nn.softmax(fc3)
    loss = tf.losses.softmax_cross_entropy(y, fc3)
    train = tf.train.AdamOptimizer().minimize(loss)
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(fc3, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return softmax_output, train, loss, accuracy
