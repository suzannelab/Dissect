import warnings
import numpy as np
import scipy as sc
from skimage.morphology import binary_dilation, dilation
import cv2
from astropy.convolution import convolve, Tophat2DKernel
from skimage import filters as filters


def z_project(image, method='max', n=None, metadata=None):
    """ Make a z projection

    Parameters
    ----------
    image: np.array
    method: str, 'max' or 'mean'
    n: int, if None, make projection on all z section,
    else take n section above and bellow the position of the maximal value

    Returns
    -------
    projection : np.array in 2D

    """

    # Verify that is 3D images
    if len(image.shape) != 3:
        warnings.warn('This is not a 3D images. The same array is returned.')
        return image

    if method == 'max':
        projection = np.max(image, axis=0)
    elif method == 'mean':
        if n is None:
            projection = np.mean(image, axis=0)
        else:
            # maybe it can be simplify
            id_max = np.argmax(image, axis=0)
            projection = np.zeros(image.shape[1:])
            l_x, l_y = np.meshgrid(range(image.shape[1:][0]),
                                   range(image.shape[1:][1]))
            for x, y in zip(l_x.ravel(), l_y.ravel()):
                min_bound = 0 if id_max[x, y] - n < 0 else id_max[x, y] - n
                max_bound = image.shape[0] if id_max[
                    x, y] + n + 1 > image.shape[0] else id_max[x, y] + n + 1
                projection[x, y] = np.nanmean(
                    image[min_bound:max_bound, x, y])
    return projection


def b_dilation(mask, width=2):
    """ Make a symetrical dilation in all direction

    Parameters
    ----------
    mask: nd.array, with the background set to 0 and the foreground to 1
    width: int, size of the dilation.
    """
    if width == 0:
        return mask
    selem = np.ones(np.repeat(2 * width + 1, len(mask.shape)))
    return (binary_dilation(mask, selem=selem)).astype(int)


def thinning(mask, width=2):
    """ Thin the mask by suppressing to the mask the result of erosion function

    Parameters
    ----------
    mask: nd.array, with the background set to 0 and the foreground to 1
    width: int, size of the erosion.
    """
    kernel = np.ones((width,width),np.uint8)
    erosion = cv2.erode(mask.astype(np.uint8), kernel, iterations = 1)

    return mask-erosion

def otsufilter(image, nbins=65536):
    """
    Apply an Otsu filter.

    Parameters
    ----------
    image: nd.array, need to be a 2D nd.array

    Returns
    -------
    mask: nd.array, where the background is set to 0 and the foreground to 1
    Change nbins with the type of images (8bits = 256, 16bits = 65536
    """
    val = filters.threshold_otsu(image, nbins=nbins)
    mask = image > val
    mask = sc.ndimage.binary_fill_holes(mask)
    return mask


def normalise_im(im, kernelsize):
    """Normalise the image by the background."""
    norm_im = im / \
        np.mean(im[np.where(~otsufilter(convolve(im, Tophat2DKernel(kernelsize))))])
    return norm_im


def apical_mask(img, face_df, edge_df, vert_df, points_df, pixel_size, it, dil):
    apical = np.zeros(img.shape)

    for i in range(len(face_df)) : 
        cell_i = enlarge_face_plane(img,
                       face_df,
                       edge_df,
                       vert_df,
                       points_df,
                       i,
                       dil,
                       {"X_SIZE":X_SIZE, 
                        "Y_SIZE":Y_SIZE, 
                        "Z_SIZE":Z_SIZE})
        
        apical[np.where(cell_i==1)]=1
        fill_apical =  ndimage.binary_closing(apical, iterations = it).astype(np.int)
    return fill_apical

def rectangle_mask(img, 
                   x_min, x_max, 
                   y_min, y_max, 
                   z_min, z_max):
    rectangle_mask = np.zeros(img.shape)
    rectangle_mask[:,:,:]=1
    rectangle_mask[z_min:z_max, y_min:y_max,  x_min:x_max]=0
    return rectangle_mask

def skel_array(self):
    """ Create a np array with field value from skeleton
    Returns
    -------
    skel_array: np.array
    """
    skel_array = np.zeros((self.specs['bbox_delta']).astype(int))
    for i, coord in self.point[list('xyz')[:self.specs['ndims']]].iterrows():
        if self.specs['ndims'] == 2:
            skel_array[coord.astype(int)[0], coord.astype(int)[
                1]] = self.point.field_value[i]
        else:
            skel_array[coord.astype(int)[0], coord.astype(int)[
                1], coord.astype(int)[2]] = self.point.field_value[i]
    return skel_array.T

def greyscale_dilation(image, width=2):
    if width == 0:
        return image
    selem = np.ones(np.repeat(2 * width + 1, len(image.shape)))
    return dilation(image, selem=selem)




