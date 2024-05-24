#!/usr/bin/env python3
"""Create Confusion"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix
    Parameters:
    -----------
    labels numpy.ndarray(m, classes):
        One-hot containing the correct labels for each data point
        m is the number of data points
        classes is the number of classes
    logits numpy.ndarray(m, classes):
        One-hot containing the predicted labels
        m is the number of data points
        classes is the number of classes
    """
    labels_num = labels.shape[1]
    confusion_matrix = np.zeros(shape=(labels_num, labels_num))
    for i in range(len(labels)):
        confusion_matrix[labels[i].nonzero()[0], logits[i].nonzero()[0]] += 1
    return confusion_matrix
