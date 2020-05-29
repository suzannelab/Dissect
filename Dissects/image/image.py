# coding: utf-8
import numpy as np
import scipy as sc
from astropy.io import fits
from astropy.convolution import convolve, Tophat2DKernel
from PIL import Image
from skimage import filters as filters
from io.io import stack_importation


def otsufilter(img):
    """
    Applie an Otsu filter.

    It takes in arguments a matrix.
    Returns a mask where the background is set to 0 and the foreground to 1
    Change nbins with the type of images (8bits = 256, 16bits = 65536
    """
    val = filters.threshold_otsu(img, nbins=65536)
    mask = img < val
    mask = np.invert(mask)
    mask = sc.ndimage.binary_fill_holes(mask)
    return mask


def proj_around_max(matrix, shape, number):
    """
    Make a maximum intensity projection of a 3D matrix.

    Parameters
    ----------
    matrix : 3D numpy.ndarray
        The 3D array of the image

    shape : tuple
        Shape of the 3D matric in a tuple (output of numpy.shape(matrix))

    number : integer
        The number of points around the max to average. The mean will be
        calculated with 2*number+1 points (middle point = the max)
    Returns
    -------
    projection : 2D numpy.ndarray
        Projected image
    """
    projection = np.zeros(shape)
    for x in range(shape[0]):
        for y in range(shape[1]):
            ind_max = np.argmax(matrix[x, y, :])
            inf = max(ind_max - number, 0)
            sup = min(ind_max + (number + 1), matrix.shape[2])
            projection[x, y] = np.mean(matrix[x, y, inf:sup])
    return projection


def tif_to_fits(base, chemin, stack=True, n=None):
    """
    Convert image.tif to image.fits.

    Use proj_aroundMAX to define the width of projection
    """
    if stack:
        im3d, im3d_size, im3d_shape = stack_importation(chemin)
        if n is None:
            raise ValueError('Error : You said TIF image is a stack need to provide keyword n')
        else:
            im = proj_around_max(im3d, im3d_shape, n)
    else:
        im = Image.open(chemin)
    hdu = fits.PrimaryHDU(im)
    hdu.writeto(base, overwrite=True)
    return


def normalise_im(im, kernelsize):
    """Normalise the image by the backgroung."""
    norm_im = im / np.mean(im[np.where(~otsufilter(convolve(im, Tophat2DKernel(kernelsize))))])
    return norm_im
