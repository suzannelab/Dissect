import numpy as np
from skimage import morphology, filters
from skimage import segmentation as ski_seg
from scipy import ndimage as ndi
from skimage.morphology import binary_dilation

def segmentation(mask, min_area=None):
    """
    Segment the cells of the image.

    Paramaters
    ----------
    mask: np.array, filament=1 and background=0
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
    markers[mask == 0] = 2
    markers[mask > 0] = 1

    segmentation = ski_seg.watershed(edges, markers)
    segmentation, _ = ndi.label(segmentation == 2)
    if min_area is not None:
        segmentation = morphology.remove_small_objects(segmentation, min_area)

    return segmentation


def junction_around_cell(mask, seg, cell):
    """Find junctions around cell i.

    Parameters
    ----------
    mask: np.array, filament=1 and background=0
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
        segmentationi.dtype) * mask)

    return juncelli

def vertices(mask, seg):

    image_vertex = np.zeros_like(mask)

    for i in range(1, np.unique(seg0)[-1]+1):


        image_cell_mask_i = np.zeros_like(mask)
        image_cell_mask_i[np.where(seg2 == i)] = 1
        seg_dilate_i = dilation(image_cell_mask_i, 1)

        image_vertex = image_vertex + seg_dilate_i

    list_vertices = np.where(image_vertex >= 3) #Récupération des vextex 'simples'


    columns_name = ['x_0',
                    'y_0',
	            'Cell_1',
		    'Cell_2',
	            'Cell_3',
	            'Cell_4',
		    'Cell_5']

    nb_vertices = len(list_vertices[0])
    init = np.zeros((nb_vertices , len(columns_name)))

    df_vertices = pd.DataFrame(data=init, columns=columns_name)

    for v in range(0, len(list_vertices[0])):
        df_vertices.loc[v]['x_0'] = list_vertices[0][v]
        df_vertices.loc[v]['y_0'] = list_vertices[1][v]

        carre = seg0[list_vertices[0][v]-3 : list_vertices[0][v]+4, list_vertices[1][v]-3 : list_vertices[1][v]+4]
        cells = np.unique(carre)

        df_vertices.loc[v]['Cell_1'] = cells[1]
        df_vertices.loc[v]['Cell_2'] = cells[2]
        df_vertices.loc[v]['Cell_3'] = cells[3]

        if len(np.unique(carre)) == 4 :
            df_vertices.loc[v]['Cell_4'] = 'Nan'
            df_vertices.loc[v]['Cell_5'] = 'Nan'
        if len(np.unique(carre)) == 5 :
            df_vertices.loc[v]['Cell_4'] = cells[4]
            df_vertices.loc[v]['Cell_5'] = 'Nan'
        if len(np.unique(carre)) == 6 :
            df_vertices.loc[v]['Cell_4'] = cells[4]
            df_vertices.loc[v]['Cell_5'] = cells[5]


    return image_vertex, list_vertices, df_vertices
