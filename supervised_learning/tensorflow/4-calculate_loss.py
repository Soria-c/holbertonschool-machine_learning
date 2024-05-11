#!/usr/bin/env python3
"""Calculate Accuracy"""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """
    calculates the softmax cross-entropy loss of a prediction
    ```
    Parameters:
    -----------
        y is a placeholder for the labels of the input data (one hot)
        y_pred is a tensor containing the network’s predictions ()
    """
    return tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
