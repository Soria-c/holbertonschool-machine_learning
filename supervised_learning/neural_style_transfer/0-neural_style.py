#!/usr/bin/env python3
"""
Neural Style Transfer
"""

import numpy as np
import tensorflow as tf


class NST:
    """
    NST Class
    """
    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Constructor
        """
        # Validate style_image
        if not isinstance(style_image, np.ndarray) or\
                style_image.shape[-1] != 3:
            raise TypeError("style_image must be a\
                     numpy.ndarray with shape (h, w, 3)")

        # Validate content_image
        if not isinstance(content_image, np.ndarray) or\
                content_image.shape[-1] != 3:
            raise TypeError("content_image must be a\
                numpy.ndarray with shape (h, w, 3)")

        # Validate alpha and beta
        if not (isinstance(alpha, (int, float)) and alpha >= 0):
            raise TypeError("alpha must be a non-negative number")
        if not (isinstance(beta, (int, float)) and beta >= 0):
            raise TypeError("beta must be a non-negative number")

        # Set instance attributes
        self.style_image = NST.scale_image(style_image)
        self.content_image = NST.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """Rescales an image to have its largest side as
            512 pixels and pixel values between 0 and 1."""
        # Validate the image shape
        if not isinstance(image, np.ndarray) or image.shape[-1] != 3:
            raise TypeError("image must be a numpy.ndarray\
                with shape (h, w, 3)")

        # Get original dimensions
        original_shape = image.shape
        h, w = original_shape[:2]

        # Determine scaling factor
        if h > w:
            scale = 512 / h
        else:
            scale = 512 / w

        # Calculate new dimensions
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize the image using bicubic interpolation
        image_resized = tf.image.resize(image,
                                        (new_h, new_w), method='bicubic')

        # Rescale pixel values to [0, 1]
        image_rescaled = image_resized / 255.0

        # Add batch dimension
        # Shape (1, h_new, w_new, 3)
        return tf.expand_dims(image_rescaled, axis=0)
