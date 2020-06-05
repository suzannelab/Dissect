# coding: utf-8

import numpy as np
from skimage.io.collection import MultiImage, Image
from astropy.io import fits
from Dissects.image import proj_around_max


def stack_importation(path):
    """
    Import a stack of images .TIF in a 3D matrix.

    Take in arguments the path of the file.
    """
    img = MultiImage(path)
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
    return stack


def import_im(path, stack=True, proj=False, N=None):
    """
    Import image for tofits.
    """
    if stack:
        im3d = stack_importation(path)
        if proj is True:
            if N is None:
                raise ValueError('Error : to project your image, provide keyword N i.e. width of projection')
            else:
                im = proj_around_max(im3d, N)
        else:
            im = im3d
    else:
        im = Image.open(path)

    print('image:', len(im.shape), 'dimensions')
    return im


def tofits(path, stack=True, proj=False, N=None):
    """
    Convert a tif file into a fits file.

    Parameters
    ----------
    path: string
        The full path (image directory + image name) of the tif image

    stack: boolean
        True if your image is a pile of stacks

    proj : boolean
        False if you want to keep a 3D image. True to project

    N: integer
        If N=0: maximum projection
        If N>0: mean of the 2N+1 pixels around the maximum
        (if N larger than (number of stack)/2: simply a mean projection)


    """
    im = import_im(path, stack, proj, N)
    hdu = fits.PrimaryHDU(im)
    # modification de l'header pour avoir la conversion µm/pixel
    #fits.setval(hdu, 'pixel_width', value='M31')
    #fits.setval(hdu, 'pixel_height', value='M31')
    #fits.setval(hdu, 'voxel_depth', value='M31')
    if proj is True:
        if N == 0:
            hdu.writeto(path.split('.')[0] + '_projMAX_' + '.fits',
                        overwrite=True)
            imfile = path.split('.')[0] + '_projMAX_' + '.fits'
            print('saved file:', imfile)
        else:
            hdu.writeto(path.split('.')[0] + '_projWidth' + str(2 * N + 1) + '.fits',
                        overwrite=True)
            imfile = path.split('.')[0] + '_projWidth' + str(2 * N + 1) + '.fits'
            print('saved file:', imfile)
    if proj is False:
        hdu.writeto(path.split('.')[0] + '_3D' + '.fits', overwrite=True)
        imfile = path.split('.')[0] + '_3D' + '.fits'
        print('saved file:', imfile)
    return imfile
