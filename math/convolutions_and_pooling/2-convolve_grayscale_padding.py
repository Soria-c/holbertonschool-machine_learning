#!/usr/bin/env python3
"""Convolution with Padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding
    ```
    Parameters:
    -----------
    images: numpy.ndarray(m, h, w)
        contains multiple grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    kernel: numpy.ndarray(kh, kw)
        contains the kernel for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    padding tuple(ph, pw)
        ph is the padding for the height of the image
        pw is the padding for the width of the image
    Returns:
    --------
    numpy.ndarray containing the convolved images
    """
    data = np.pad(images, pad_width=((0, 0),
                                     (padding[0], padding[0]),
                                     (padding[1], padding[1])))
    input_dimensions = data.shape
    kernel_size_v = kernel.shape[0]
    kernel_size_h = kernel.shape[1]
    n_images = input_dimensions[0]
    fm_v = input_dimensions[1] - kernel_size_v + 1
    fm_h = input_dimensions[2] - kernel_size_h + 1
    map_dimensions = (n_images, fm_v, fm_h)
    feature_maps = np.zeros(shape=map_dimensions)
    for h_index in range(fm_h):
        for v_index in range(fm_v):
            slice_2d = data[:, v_index:kernel_size_v+v_index,
                            h_index:kernel_size_h+h_index]
            feature_maps[:, v_index, h_index] = np.sum(slice_2d
                                                       * kernel, axis=(1, 2))
    return feature_maps
