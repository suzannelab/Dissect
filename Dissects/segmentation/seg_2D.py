
# comment
import numpy as np
from astropy.convolution import convolve, Box2DKernel
from skimage import morphology, filters
import scipy
from scipy import ndimage as ndi


def segmentation(inv_maskfil, min_area=None):
    """
    Segment the cells of the image.

    Paramaters
    ----------
    inv_maskfil: np.array
        The mask filaments = 0 and cells = 1
    mean_area: integer
        minimum number of pixels of a cell
    Return
    ------
    segmentation: np.array
        Pixels of filaments are equal to 0
        Pixels of the background = 1
        Pixels of cell i = i
    """
    edges = filters.sobel(inv_maskfil)
    markers = np.zeros_like(inv_maskfil)
    markers[inv_maskfil == 0] = 1
    markers[inv_maskfil > 0] = 2

    segmentation = morphology.watershed(edges, markers)
    segmentation, _ = ndi.label(segmentation == 2)
    if min_area is not None:
        segmentation = morphology.remove_small_objects(segmentation, min_area)

    return segmentation


def junction_around_cell(maskfil, seg, cell):
    """Find junctions around cell i.

    Parameters
    ----------
    maskfil: np.array
        filament = 1, cells and background = 0
    seg: np.array
        output of the segmentation function
    cell: integer
        number of the chosen cell

    Returns
    -------
    juncelli: np.array
        background = 0, one-pixel-width junction around cell i = 1

    """
    segmentationi = np.zeros_like(seg)
    segmentationi[np.where(seg == cell)] = 1
    '''
    # Box smooth around unique cell + multiply by MaskFil to have pixel
    # filaments
    width=2
    kernel=Box2DKernel(width)
    JuncCelli=(convolve(segmentationi, kernel) *
                 maskfil).astype(bool).astype(int)
    '''

    juncelli = scipy.ndimage.binary_dilation(segmentationi).astype(segmentationi.dtype) * maskfil

    return juncelli
