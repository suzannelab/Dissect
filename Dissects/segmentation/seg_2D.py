import numpy as np
import pandas as pd
from skimage import morphology, filters
from skimage import segmentation as ski_seg
from scipy import ndimage as ndi
from skimage.morphology import binary_dilation
from Dissects.image import thinning
from Dissects.image import dilation


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


def vertices(mask, seg=None, dilation_width=1):
    """Find the vertices (nodes) of the skeleton

    Parameters
    ----------
    mask: np.array, filament=1 and background=0

    Returns
    -------
    image_vertex: np.array where cells=1, junctions=2, vertices >=3
    list_vertices: list of the vertices
    df_vertices: dataframe with for each vertex x_0: x coordinate of the vertex
                                                y_0: y coordinate of the vertex
                                                Cell_i: the cells connected to the vertex


    """

    if seg is None:
        seg = segmentation(mask)

    image_vertex = np.zeros_like(mask)

    for i in range(1, np.unique(seg)[-1] + 1):
        image_cell_mask_i = np.zeros_like(mask)
        image_cell_mask_i[np.where(seg == i)] = 1
        seg_dilate_i = dilation(image_cell_mask_i, dilation_width)
        image_vertex = image_vertex + seg_dilate_i
    list_vertices = np.where(image_vertex >= 3)  # Récupération des vextex 'simples'

    columns_name = ['x_0',
                    'y_0',
                    'Cell_1',
                    'Cell_2',
                    'Cell_3',
                    'Cell_4',
                    'Cell_5']

    init = np.zeros((len(list_vertices[0]), len(columns_name)))

    df_vertices = pd.DataFrame(data=init, columns=columns_name)

    for v in range(0, len(list_vertices[0])):
        df_vertices.loc[v]['x_0'] = list_vertices[0][v]
        df_vertices.loc[v]['y_0'] = list_vertices[1][v]

        carre = seg[list_vertices[0][v] - 3: list_vertices[0][v] +
                    4, list_vertices[1][v] - 3: list_vertices[1][v] + 4]
        cells = np.unique(carre)

        df_vertices.loc[v]['Cell_1'] = cells[1]
        df_vertices.loc[v]['Cell_2'] = cells[2]
        df_vertices.loc[v]['Cell_3'] = cells[3]

        if len(np.unique(carre)) == 4:
            df_vertices.loc[v]['Cell_4'] = 'Nan'
            df_vertices.loc[v]['Cell_5'] = 'Nan'
        if len(np.unique(carre)) == 5:
            df_vertices.loc[v]['Cell_4'] = cells[4]
            df_vertices.loc[v]['Cell_5'] = 'Nan'
        if len(np.unique(carre)) == 6:
            df_vertices.loc[v]['Cell_4'] = cells[4]
            df_vertices.loc[v]['Cell_5'] = cells[5]

    ind = 0
    while ind < len(df_vertices):
        cells_ind = np.array([df_vertices['Cell_1'].iloc[ind],
                              df_vertices['Cell_2'].iloc[ind],
                              df_vertices['Cell_3'].iloc[ind],
                              df_vertices['Cell_4'].iloc[ind],
                              df_vertices['Cell_5'].iloc[ind]])

        liste_x0 = [df_vertices['x_0'].iloc[ind]]
        liste_y0 = [df_vertices['y_0'].iloc[ind]]
        liste_i = []
        liste_cellsi = [cells_ind]

        for i in range(ind + 1, len(df_vertices)):

            cells_i = np.array([df_vertices['Cell_1'].iloc[i],
                                df_vertices['Cell_2'].iloc[i],
                                df_vertices['Cell_3'].iloc[i],
                                df_vertices['Cell_4'].iloc[i],
                                df_vertices['Cell_5'].iloc[i]])

            mask_TrueFalse = np.isin(cells_ind, cells_i)

            if np.sum(mask_TrueFalse) >= 3:
                liste_cellsi.append(cells_i)
                liste_i.append(df_vertices.axes[0][i])
                liste_x0.append(df_vertices['x_0'].iloc[i])
                liste_y0.append(df_vertices['y_0'].iloc[i])

        df_vertices['x_0'].iloc[ind] = int(np.round(np.mean(liste_x0)))
        df_vertices['y_0'].iloc[ind] = int(np.round(np.mean(liste_y0)))

        ucells = np.unique(liste_cellsi)

        try:
            df_vertices['Cell_1'].iloc[ind] = ucells[0]
            df_vertices['Cell_2'].iloc[ind] = ucells[1]
            df_vertices['Cell_3'].iloc[ind] = ucells[2]
            df_vertices['Cell_4'].iloc[ind] = ucells[3]
            df_vertices['Cell_5'].iloc[ind] = ucells[4]
        except:
            pass

        # print(liste_i)
        df_vertices = df_vertices.drop(liste_i)

        ind += 1

    df_vertices.reset_index(drop=True, inplace=True)
    return image_vertex, list_vertices, df_vertices


def junctions(list_vertices, df_vertices):
    """Create a dataframe of the cell junctions

    Parameters
    ----------
    list_vertices: list of the vertices given by the 'vertices' function
    df_vertices: dataframe given by the 'vertices' function

    Returns
    -------
    df_junction: dataframe with Cell1: first cell of the junction
                                Cell2: second cell of the junction
                                x0: x coordinate of the startpoint of the junction
                                y0: y coordinate of the startpoint of the junction
                                x1: x coordinate of the endpoint of the junction
                                y1: y coordinate of the endpoint of the junction
                                angle: angle between the horozintal plane and the junction
                                lenght: lenght of the junction
    """

    df4_jonctions = pd.DataFrame(columns=['Cell1',
                                          'Cell2',
                                          'x0',
                                          'y0',
                                          'x1',
                                          'y1',
                                          ])

    for ind in range(0, df_vertices.shape[0]):  # pour chaque vertex

        cells_ind = np.array([df_vertices['Cell_1'][ind],
                              df_vertices['Cell_2'][ind],
                              df_vertices['Cell_3'][ind],
                              df_vertices['Cell_4'][ind],
                              df_vertices['Cell_5'][ind]])

        for i in range(ind + 1, df_vertices.shape[0]):  # pour chaque autre vertex

            cells_i = np.array([df_vertices['Cell_1'][i],
                                df_vertices['Cell_2'][i],
                                df_vertices['Cell_3'][i],
                                df_vertices['Cell_4'][i],
                                df_vertices['Cell_5'][i]])

            mask_TrueFalse = np.isin(cells_ind, cells_i)

            if np.isin(1., cells_ind):

                if np.sum(mask_TrueFalse) == 3:

                    dict_jonctions = {'Cell1': cells_ind[np.where(mask_TrueFalse)][0],
                                      'Cell2': cells_ind[np.where(mask_TrueFalse)][1],
                                      'x0': df_vertices['x_0'][i],
                                      'y0': df_vertices['y_0'][i],
                                      'x1': df_vertices['x_0'][ind],
                                      'y1': df_vertices['y_0'][ind],
                                      }

                    df4_jonctions = df4_jonctions.append(
                        dict_jonctions, ignore_index=True)

            else:

                if np.sum(mask_TrueFalse) >= 2:

                    dict_jonctions = {'Cell1': cells_ind[np.where(mask_TrueFalse)][0],
                                      'Cell2': cells_ind[np.where(mask_TrueFalse)][1],
                                      'x0': df_vertices['x_0'][i],
                                      'y0': df_vertices['y_0'][i],
                                      'x1': df_vertices['x_0'][ind],
                                      'y1': df_vertices['y_0'][ind],
                                      }

                    df4_jonctions = df4_jonctions.append(
                        dict_jonctions, ignore_index=True)
    return df4_jonctions


def generate_mesh(mask, seg=None, dilation_width=1):
    """ Generate mesh

    Parameters
    ----------
    mask: np.array, filament=1 and background=0

    Returns
    -------
    face_df
    edge_df
    vert_df

    """

    # Vertices detection
    image_vertex, list_vertices, df_vertices = vertices(mask, seg, dilation_width)

    # Edge detection
    df_junctions = junctions(list_vertices, df_vertices)

    vert_df = pd.DataFrame(columns=list('xy'))
    edge_df = pd.DataFrame(columns=['cell1',
                                    'cell2',
                                    'v1',
                                    'v2'])

    # change data output
    for _, e in df_junctions.iterrows():

        edge = {'cell1': e.Cell1,
                'cell2': e.Cell2}

        df_ = vert_df[vert_df['x'] == e.x0]
        if df_.empty:
            vert_df = vert_df.append({'x': e.x0,
                                      'y': e.y0},
                                     ignore_index=True)
            edge['v1'] = vert_df.index[-1]
        else:
            df_ = df_[df_['y'] == e.y0]
            if df_.empty:
                vert_df = vert_df.append({'x': e.x0,
                                          'y': e.y0},
                                         ignore_index=True)
                edge['v1'] = vert_df.index[-1]
            else:
                edge['v1'] = df_.index[0]

        df_ = vert_df[vert_df['x'] == e.x1]
        if df_.empty:
            vert_df = vert_df.append({'x': e.x1,
                                      'y': e.y1},
                                     ignore_index=True)
            edge['v2'] = vert_df.index[-1]
        else:
            df_ = df_[df_['y'] == e.y1]
            if df_.empty:
                vert_df = vert_df.append({'x': e.x1,
                                          'y': e.y1},
                                         ignore_index=True)
                edge['v2'] = vert_df.index[-1]
            else:
                edge['v2'] = df_.index[0]

        edge_df = edge_df.append(edge,
                                 ignore_index=True)

    edge_df['cell1'] = pd.to_numeric(edge_df['cell1'], downcast='integer')
    edge_df['cell2'] = pd.to_numeric(edge_df['cell2'], downcast='integer')
    edge_df['v1'] = pd.to_numeric(edge_df['v1'], downcast='integer')
    edge_df['v2'] = pd.to_numeric(edge_df['v2'], downcast='integer')

    vert_df['x'] = pd.to_numeric(vert_df['x'], downcast='integer')
    vert_df['y'] = pd.to_numeric(vert_df['y'], downcast='integer')

    face_id = np.sort(np.unique(np.concatenate((edge_df.cell1.sort_values().to_numpy(),
                                                edge_df.cell2.sort_values().to_list()))))
    face_df = pd.DataFrame(index=face_id, columns=list('xy'))

    for f in face_id:

        v_id = np.concatenate((edge_df[(edge_df['cell1'] == f) | (edge_df['cell2'] == f)]['v1'].to_numpy(),
                               edge_df[(edge_df['cell1'] == f) | (edge_df['cell2'] == f)]['v2'].to_numpy()))

        v_id = np.unique(v_id)

        face_df.loc[f]['x'] = vert_df.loc[v_id].x.mean()
        face_df.loc[f]['y'] = vert_df.loc[v_id].y.mean()

    return face_df, edge_df, vert_df
