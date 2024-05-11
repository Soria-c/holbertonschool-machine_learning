#!/usr/bin/env python3
"""Calculate Accuracy"""
import tensorflow.compat.v1 as tf


create_layer = __import__('1-create_layer').create_layer


def calculate_accuracy(y, y_pred):
    """
    Function that creates the forward propagation graph for the neural network
    ```
    Parameters:
    -----------
        y is a placeholder for the labels of the input data
        y_pred is a tensor containing the network’s predictions
    """
    labels = tf.math.argmax(y, axis=1)
    prediction = tf.math.argmax(y_pred, axis=1)
    correct_predictions_encoded = tf.cast(tf.math.equal(
        labels, prediction), tf.float32)
    return tf.math.reduce_mean(correct_predictions_encoded)
