import itertools

import networkx as nx
import numpy as np
import pandas as pd

from io import StringIO
from scipy import ndimage
from sklearn.neighbors import KDTree, BallTree

from ..utils.utils import pixel_to_um


default_image_specs = {
    "X_SIZE": 1,
    "Y_SIZE": 1,
    "Z_SIZE": 1
}


def generate_segmentation(skeleton,
                          clean=True,
                          **kwargs
                          ):
    """
    Generate three dataframe from binary skeleton image which contains informations
    about faces, edges and vertex
    Put coordinates of vertex into um. Keep pixel position in "xyz_pix" columns. 

    Parameters
    ----------
    skeleton : binary np.array; background=0, skeleton=1

    Return
    ------
    face_df : DataFrame
    edge_df : DataFrame
    vert_df : DataFrame
    """
    image_specs = default_image_specs
    image_specs.update(**kwargs)

    vert_df = find_vertex(skeleton)
    vert_df, edge_df = find_edge(skeleton, vert_df, half_edge=True)
    face_df, edge_df = find_cell(edge_df)

    points_df = find_points(edge_df)
    
    vert_df['x_pix'] = vert_df['x']
    vert_df['y_pix'] = vert_df['y']
    vert_df['z_pix'] = vert_df['z']

    pixel_to_um(vert_df, image_specs, ['x_pix', 'y_pix', 'z_pix'], list('xyz'))
    pixel_to_um(points_df, image_specs, ['x_pix', 'y_pix', 'z_pix'], list('xyz'))

    
    return face_df, edge_df, vert_df, points_df


def find_points(edge_df):
    points_df = pd.DataFrame(
        columns=['x_pix', 'y_pix', 'z_pix', 'edge', 'face'])

    for e, val in edge_df.iterrows():
        for i in range(len(val['point_x'][0])):
            dict_ = {'x_pix': val['point_x'][0][i],
                     'y_pix': val['point_y'][0][i],
                     'z_pix': val['point_z'][0][i],
                     'edge': e,
                     'face': val.face}
            points_df = points_df.append(dict_,ignore_index=True)

    return points_df


def find_vertex(skeleton):
    save_column = list('xyz')[:skeleton.specs['ndims']]
    save_column.append('nfil')
    vert_df = skeleton.critical_point[skeleton.critical_point.nfil >= 3][save_column]
    return vert_df


def find_edge(skeleton, vert_df, half_edge=True):

    # find edges
    edge_df = pd.DataFrame(dtype='int')

    for i, val in vert_df.iterrows():
        start_cps = np.unique(skeleton.filament[(skeleton.filament.cp1 == i) | (
            skeleton.filament.cp2 == i)][['cp1', 'cp2']])
        for start in start_cps:
            sc = start

            filaments_id = []
            if sc != i:
                # Get the first filament portion
                try:
                    filaments_id.append(skeleton.filament[(skeleton.filament.cp1 == i) | (skeleton.filament.cp2 == i) &
                                                          (skeleton.filament.cp1 == sc) | (skeleton.filament.cp2 == sc)].index[0])
                except:
                    pass
                previous_sc = i
                previous_previous_sc = previous_sc
                previous_sc = sc
                while skeleton.critical_point.loc[sc]['nfil'] < 3:
                    tmp_sc = np.unique(skeleton.filament[(skeleton.filament.cp1 == previous_sc) | (
                        skeleton.filament.cp2 == previous_sc)][['cp1', 'cp2']])

                    for sc in tmp_sc:

                        if (sc != previous_previous_sc) and (sc != previous_sc):

                            try:
                                filaments_id.append(skeleton.filament[(skeleton.filament.cp1 == previous_sc) | (skeleton.filament.cp2 == previous_sc) &
                                                                      (skeleton.filament.cp1 == sc) | (skeleton.filament.cp2 == sc)].index[0])
                            except:
                                pass
                            previous_previous_sc = previous_sc
                            previous_sc = sc
                            break

                # Get coordinates from filament ids
                pixel_x = skeleton.point[skeleton.point.filament.isin(
                    filaments_id)]['x'].to_numpy()
                pixel_y = skeleton.point[skeleton.point.filament.isin(
                    filaments_id)]['y'].to_numpy()
                pixel_z = skeleton.point[skeleton.point.filament.isin(
                    filaments_id)]['z'].to_numpy()
#                 print(pixel_x)
                edges = {'srce': i,
                         'trgt': sc,
                         'point_x': pixel_x,
                         'point_y': pixel_y,
                         'point_z': pixel_z}
                edge_df = edge_df.append(edges, ignore_index=True)

    edge_df.drop(edge_df[edge_df.srce == edge_df.trgt].index, inplace=True, )
    edge_df['min'] = np.min(edge_df[['srce', 'trgt']], axis=1)
    edge_df['max'] = np.max(edge_df[['srce', 'trgt']], axis=1)
    edge_df['srce'] = edge_df['min']
    edge_df['trgt'] = edge_df['max']
    edge_df.drop(['min', 'max'], axis=1, inplace=True)
    edge_df.drop_duplicates(inplace=True, subset=['srce', 'trgt'])
    edge_df.reset_index(drop=True, inplace=True)

    if half_edge:
        edge_df = generate_half_edge(edge_df, vert_df)
    return vert_df, edge_df


def clean_vertex_image(vertex_image):
    """
    Group vertex into one if there is vertex detected too close to each other
    Group vertex if they are closed, meaning connected in any axis. 

    Parameters
    ----------
    vertex_image: binary image with 1 where there is a vertex

    Return
    ------
    vert_df: DataFrame of vertex 
    """
    s = ndimage.generate_binary_structure(3, 3)
    labeled_array, num_features = ndimage.label(vertex_image, structure=s)
    unique_, count_ = np.unique(labeled_array, return_counts=True)

    vertex = {}

    index = 0
    for u, c, in zip(unique_, count_):
        if c == 1:
            vertex[index] = np.array(np.where(labeled_array == u)).flatten()
        else:
            vertex[index] = np.mean(
                np.array(np.where(labeled_array == u)), axis=1, dtype='int')

        index += 1

    vert_df = pd.DataFrame.from_dict(
        vertex, orient='index', columns=list('zyx'))

    # remove first line which is the background
    vert_df = vert_df.loc[1:]

    return vert_df


def generate_half_edge(edge_df, vert_df):
    """
    Generate half edge dataframe from edge and vert data frame

    Parameters
    ----------
    edge_df : DataFrame
    vert_df : DataFrame

    Return
    ------
    new_edge_df : DataFrame of half edge
    """
    new_edge_df = pd.DataFrame(
        data=[np.zeros(len(edge_df.columns))], columns=edge_df.columns, dtype=object)
    for v0, data in vert_df.iterrows():
        va = [edge_df[(edge_df.srce == v0)]['trgt'].to_numpy()]
        va.append(edge_df[(edge_df.trgt == v0)]['srce'].to_numpy())
        va = [item for sublist in va for item in sublist]
        for v in va:
            dict_ = {'srce': v0,
                     'trgt': v,
                     }
            for c in edge_df.columns:
                if (c != 'srce') & (c != 'trgt'):
                    dict_[c] = edge_df[
                        ((edge_df.srce == v0) & (edge_df.trgt == v))
                        |
                        (edge_df.trgt == v0) & (edge_df.srce == v)
                    ][c].to_numpy()

            new_edge_df.loc[
                np.max(new_edge_df.index)+1
            ] = dict_

            dict_ = {'srce': v,
                     'trgt': v0,
                     }
            for c in edge_df.columns:
                if (c != 'srce') & (c != 'trgt'):
                    dict_[c] = edge_df[
                        ((edge_df.srce == v0) & (edge_df.trgt == v))
                        |
                        (edge_df.trgt == v0) & (edge_df.srce == v)
                    ][c].to_numpy()

            new_edge_df.loc[
                np.max(new_edge_df.index)+1
            ] = dict_

    new_edge_df.drop(index=0, axis=0, inplace=True)
    new_edge_df.drop_duplicates(inplace=True, subset=['srce', 'trgt'])
    new_edge_df.reset_index(drop=True, inplace=True)

    return new_edge_df


def find_cell(edge_df):
    """
    Find face with graph theory

    Parameters
    ----------
    edge_df : DataFrame of edges

    Return
    ------
    face_df : DataFrame of faces
    edge_df : DataFrame of ordered edges
    """
    G = nx.from_pandas_edgelist(edge_df,
                                source='srce',
                                target='trgt',
                                create_using=nx.Graph())
    all_faces = nx.minimum_cycle_basis(G)

    order_faces = [all_faces[0]]
    all_faces.remove(all_faces[0])

    find = True
    while (len(all_faces) > 0) or (find == False):
        find = False
        for i in range(len(all_faces)):
            for j in range(len(order_faces)):
                if len(set(order_faces[j]).intersection(all_faces[i])) > 0:
                    order_faces.append(all_faces[i])
                    all_faces.remove(all_faces[i])
                    find = True
                    break
            if find == True:
                break

        if not find:
            break

    edge_df['face'] = -1
    edge_df['orient'] = -1

    cpt_face = 1

    for f in order_faces:

        edges = edge_df[(edge_df.srce.isin(f)) & (edge_df.trgt.isin(f))]

        vert_order = [edges.iloc[0].srce]
        vert_order.append(edges.iloc[0].trgt)
        for i in range(len(edges)):
            vert_order.append(edges[(edges.srce == vert_order[-1]) &
                                    (edges.trgt != vert_order[-2])]['trgt'].to_numpy()[0])
            if vert_order[0] == vert_order[-1]:
                break

        edge = []
        for i in range(len(vert_order)-1):
            edge.append(edge_df[(edge_df.srce == vert_order[i]) & (
                edge_df.trgt == vert_order[i+1])].index.to_numpy()[0])

        if len(np.unique(edge_df.loc[edge]['face'].to_numpy())) == 1:
            edge_df.loc[edge, 'face'] = cpt_face

        else:
            vert_order = np.flip(vert_order)
            edge = []
            for i in range(len(vert_order)-1):
                edge.append(edge_df[(edge_df.srce == vert_order[i]) & (
                    edge_df.trgt == vert_order[i+1])].index.to_numpy()[0])
            if len(np.unique(edge_df.loc[edge]['face'].to_numpy())) == 1:
                edge_df.loc[edge, 'face'] = cpt_face
            else:
                print("there is a problem")

        cpt_face += 1

    face_df = pd.DataFrame(index=np.sort(edge_df.face.unique()))
    return face_df, edge_df
