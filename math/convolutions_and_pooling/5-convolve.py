#!/usr/bin/env python3
"""Multiple Kernels"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels
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
    kernel_size_v = kernels.shape[0]
    kernel_size_h = kernels.shape[1]
    if (padding == "valid"):
        padding = (0, 0)
    elif (padding == "same"):
        padding = (1 + (stride[0] * (images.shape[1] - 1) -
                   images.shape[1] + kernel_size_v) // 2,
                   1 + (stride[1] * (images.shape[2] - 1) -
                   images.shape[2] + kernel_size_h) // 2)
    data = np.pad(images, pad_width=((0, 0),
                                     (padding[0], padding[0]),
                                     (padding[1], padding[1]),
                                     (0, 0)))
    input_dimensions = data.shape

    n_images = input_dimensions[0]
    fm_v = ((images.shape[1] - kernel_size_v +
             (2 * padding[0])) // stride[0]) + 1
    fm_h = ((images.shape[2] - kernel_size_h +
             (2 * padding[1])) // stride[1]) + 1
    map_dimensions = (n_images, fm_v, fm_h, kernels.shape[3])
    feature_maps = np.zeros(shape=map_dimensions)
    for k in range(kernels.shape[3]):
        for h_index in range(fm_h):
            for v_index in range(fm_v):
                slice_2d = data[:, (v_index*stride[0]):
                                kernel_size_v+(v_index*stride[0]),
                                (h_index*stride[1]):
                                kernel_size_h+(h_index*stride[1])]
                feature_maps[:, v_index, h_index, k] = \
                    np.sum(slice_2d * kernels[:, :, :, k], axis=(1, 2, 3))
    return feature_maps
