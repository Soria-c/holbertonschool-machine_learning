#!/usr/bin/env python3
"""Create placeholder"""
import tensorflow.compat.v1 as tf
# import tensorflow
# tf = tensorflow.compat.v1


def create_placeholders(nx, classes):
    """Function that returns two placeholders, x and y,
    for the neural network"""
    return tf.placeholder(dtype=tf.float32, shape=[-1, nx], name="x"),\
        tf.placeholder(dtype=tf.float32, shape=(None, classes), name="y")
