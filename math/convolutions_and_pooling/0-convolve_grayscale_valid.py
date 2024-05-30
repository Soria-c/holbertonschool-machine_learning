#!/usr/bin/env python3
"""Valid Convolution"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images
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
    Returns:
    --------
    numpy.ndarray containing the convolved images
    """
    input_dimensions = images.shape
    kernel_size = kernel.shape[0]
    n_images = input_dimensions[0]
    fm_v = input_dimensions[1] - kernel_size + 1
    fm_h = input_dimensions[2] - kernel_size + 1
    map_dimensions = (n_images, fm_v, fm_h)
    feature_maps = np.zeros(shape=map_dimensions)
    for i in range(n_images):
        for j in range(fm_v * fm_h):
            h_index = j % fm_h
            v_index = j // fm_h
            slice_2d = images[i][v_index:kernel_size+v_index,
                                 h_index:kernel_size+h_index]
            feature_maps[i][v_index, h_index] = np.sum(slice_2d * kernel)
    return feature_maps
