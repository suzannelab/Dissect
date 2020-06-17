# coding: utf-8
import numpy as np
import scipy as sc
from astropy.io import fits
from astropy.convolution import convolve, Tophat2DKernel
from PIL import Image
from skimage import filters as filters



def otsufilter(img, nbins=65536):
    """
    Applie an Otsu filter.

    It takes in arguments a matrix.
    Returns a mask where the background is set to 0 and the foreground to 1
    Change nbins with the type of images (8bits = 256, 16bits = 65536
    """
    val = filters.threshold_otsu(img, nbins=nbins)
    mask = img < val
    mask = np.invert(mask)
    mask = sc.ndimage.binary_fill_holes(mask)
    return mask


def proj_around_max(matrix, n):
    """
    Make a maximum intensity projection of a 3D matrix.

    Parameters
    ----------
    matrix : 3D numpy.ndarray
        The 3D array of the image

    n : integer
        The number of points around the max to average. The mean will be
        calculated with 2*number+1 points (middle point = the max)
    Returns
    -------
    projection : 2D numpy.ndarray
        Projected image
    """
    shape = matrix.shape
    projection = np.zeros((shape[0],shape[1]))
    l_x, l_y = np.meshgrid(range(shape[0]), range(shape[1]))
    for x, y in zip(l_x.ravel(), l_y.ravel()):
        ind_max = np.argmax(matrix[x, y, :])
        inf = max(ind_max - n, 0)
        sup = min(ind_max + (n + 1), matrix.shape[2])
        projection[x, y] = np.mean(matrix[x, y, inf:sup])
    return projection





def normalise_im(im, kernelsize):
    """Normalise the image by the backgroung."""
    norm_im = im / np.mean(im[np.where(~otsufilter(convolve(im, Tophat2DKernel(kernelsize))))])
    return norm_im
