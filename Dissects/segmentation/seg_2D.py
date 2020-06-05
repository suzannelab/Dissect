import numpy as np
from astropy.convolution import convolve, Box2DKernel
from skimage import morphology, filters
from scipy import ndimage as ndi


def segmentation(inv_maskfil, min_area=None, reverse=True):
    """

    """
    if reverse is False:
        inv_maskfil = (inv_maskfil.astype(int) - 1) * -1
    edges = filters.sobel(inv_maskfil)
    markers = np.zeros_like(inv_maskfil
    markers[inv_maskfil == 0]=1
    markers[inv_maskfil > 0]=2

    segmentation=morphology.watershed(edges, markers)
    segmentation, _=ndi.label(segmentation == 2)
    if min_area is not None : 
        segmentation = morphology.remove_small_objects(segmentation, min_area)

    return segmentation


def junction_around_cell(seg, maskfil, i, width=2):
    # trouve les jonctions autour de la cellule i
    segmentationi=np.zeros_like(seg)
    # for each cell get contour pixels
    segmentationi[np.where(seg == i)]=1

    # Box smooth around unique cell + multiply by MaskFil to have pixel
    # filaments
    kernel=Box2DKernel(width)
    JuncCelli=(convolve(segmentationi, kernel) *
                 maskfil).astype(bool).astype(int)

    return JuncCelli
