import numpy as np
from skimage import morphology, filters, segmentation
from scipy import ndimage as ndi


def segmentation(mask, min_area=None):
    """
    Segment the cells of the image.

    Paramaters
    ----------
    mask: np.array, filament=0 and background=1
    mean_area: integer, minimum number of pixels of a cell
    Return
    ------
    segmentation: np.array
        Pixels of filaments are equal to 0
        Pixels of the background = 1
        Pixels of cell i = i
    """
    edges = filters.sobel(mask)
    markers = np.zeros_like(mask)
    markers[mask == 0] = 1
    markers[mask > 0] = 2

    segmentation = segmentation.watershed(edges, markers)
    segmentation, _ = ndi.label(segmentation == 2)
    if min_area is not None:
        segmentation = morphology.remove_small_objects(segmentation, min_area)

    return segmentation


def junction_around_cell(mask, seg, cell):
    """Find junctions around cell i.

    Parameters
    ----------
    maskfil: np.array, filament=0 and background=1
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

    juncelli = (ndi.binary_dilation(segmentationi).astype(
        segmentationi.dtype) * np.invert(mask))

    return juncelli
