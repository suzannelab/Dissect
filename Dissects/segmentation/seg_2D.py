import numpy as np

from astropy.convolution import convolve, Tophat2DKernel, Box2DKernel

from skimage import morphology, filters
from scipy import ndimage as ndi


def segmentation(InvMaskFil, min_area=2, reverse=True):
    if reverse == False:
        InvMaskFil = (InvMaskFil.astype(int) - 1) * -1
    edges = filters.sobel(InvMaskFil)
    markers = np.zeros_like(InvMaskFil)
    markers[InvMaskFil == 0] = 1
    markers[InvMaskFil > 0] = 2

    segmentation = morphology.watershed(edges, markers)
    segmentation, _ = ndi.label(segmentation == 2)
    segmentation = morphology.remove_small_objects(segmentation, min_area)

    return segmentation


def junction_around_cell(seg, MaskFil, i, width=2):
    # trouve les jonctions autour de la cellule i
    segmentationi = np.zeros_like(seg)
    # for each cell get contour pixels
    segmentationi[np.where(seg == i)] = 1

    # Box smooth around unique cell + multiply by MaskFil to have pixel
    # filaments
    kernel = Box2DKernel(width)
    JuncCelli = (convolve(segmentationi, kernel) *
                 MaskFil).astype(bool).astype(int)

    return JuncCelli
