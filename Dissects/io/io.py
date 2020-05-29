# coding: utf-8

import numpy as np
from skimage.io.collection import MultiImage


def stack_importation(chemin):
    """
    Import a stack of images .TIF in a 3D matrix.

    Take in arguments the path of the folder.
    """
    img = MultiImage(chemin)
    size = len(img)
    if (size != 1):
        shape = img[0].shape
        stack = np.zeros((shape[0], shape[1], size))
        for i in range(size):
            stack[:, :, i] = img[i]
    if (size == 1):
        shape = img[0].shape
        stack = np.zeros((shape[1], shape[2], shape[0]))
        for i in range(shape[0]):
            stack[:, :, i] = img[0][i, :, :]
        shape = stack[:, :, 0].shape
    return stack, size, shape
