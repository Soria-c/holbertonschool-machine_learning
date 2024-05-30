#!/usr/bin/env python3
"""Same Convolution"""
import numpy as np


def convolve_grayscale_same(images, kernel):
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
    kernel_size = kernel.shape[0]
    padding = (kernel_size - 1) // 2
    data = np.pad(images, pad_width=((0, 0),
                                     (padding, padding), (padding, padding)))
    input_dimensions = data.shape
    n_images = input_dimensions[0]
    fm_v = input_dimensions[1] - kernel_size + 1
    fm_h = input_dimensions[2] - kernel_size + 1
    map_dimensions = (n_images, fm_v, fm_h)
    feature_maps = np.zeros(shape=map_dimensions)
    for h_index in range(fm_h):
        for v_index in range(fm_v):
            slice_2d = data[:, v_index:kernel_size+v_index,
                            h_index:kernel_size+h_index]
            feature_maps[:, v_index, h_index] = np.sum(slice_2d
                                                       * kernel, axis=(1, 2))
    return feature_maps
