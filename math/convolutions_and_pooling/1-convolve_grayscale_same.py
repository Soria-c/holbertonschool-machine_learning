#!/usr/bin/env python3
"""Same Convolution"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images
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
    kernel_size_v = kernel.shape[0]
    kernel_size_h = kernel.shape[1]
    if kernel_size_v % 2 and kernel_size_h % 2:
        padding_v = (kernel_size_v - 1) // 2
        padding_h = (kernel_size_h - 1) // 2
    else:
        padding_v = kernel_size_v // 2
        padding_h = kernel_size_h // 2
    data = np.pad(images, pad_width=((0, 0),
                                     (padding_v, padding_v),
                                     (padding_h, padding_h)))
    input_dimensions = data.shape
    n_images = input_dimensions[0]
    print(kernel_size_h, kernel_size_v)
    fm_v = images.shape[1]
    fm_h = images.shape[2]
    map_dimensions = (n_images, fm_v, fm_h)
    print(map_dimensions)
    feature_maps = np.zeros(shape=map_dimensions)
    for h_index in range(fm_h):
        for v_index in range(fm_v):
            slice_2d = data[:, v_index:kernel_size_v+v_index,
                            h_index:kernel_size_h+h_index]
            feature_maps[:, v_index, h_index] = np.sum(slice_2d
                                                       * kernel, axis=(1, 2))
    return feature_maps
