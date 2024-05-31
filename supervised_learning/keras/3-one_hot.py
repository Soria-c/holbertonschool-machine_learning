#!/usr/bin/env python3
"""One Hot"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix
    ```
    Parameters:
    -----------
    labels:
    classes:
    """
    return K.utils.to_categorical(labels, classes)
