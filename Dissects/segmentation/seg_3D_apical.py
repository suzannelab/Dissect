import itertools
import logging 
import networkx as nx
import numpy as np
import pandas as pd

from io import StringIO
from scipy import ndimage
from sklearn.neighbors import KDTree, BallTree

from ..utils.utils import pixel_to_um

logger = logging.getLogger(name=__name__)
MAX_ITER = 10


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
    pixel_to_um(points_df, image_specs, [
                'x_pix', 'y_pix', 'z_pix'], list('xyz'))

    # Mark face in the border 
    # It remove more cell than expected...
    edge_df['opposite'] = -1
    for e, val in edge_df.iterrows():
        tmp = edge_df[(edge_df.srce==val.trgt) & (edge_df.trgt== val.srce)].index.to_numpy()
        if len(tmp)>0:
            edge_df.loc[e, 'opposite'] = tmp
            
    face_df["border"] = 0
    face_df.loc[edge_df[edge_df.opposite==-1]['face'].to_numpy(), 'border'] = 1


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
            points_df = points_df.append(dict_, ignore_index=True)

    return points_df


def find_vertex(skeleton):
    """ 
    Extract vertex from DisperSE skeleton. 
    As we use output of DisperSE for determined vertices. You need to process the breakdown 
    part to have the correct placement of vertices. 
    See http://www2.iap.fr/users/sousbie/web/html/index55a0.html?category/Quick-start for more information

    Parameters
    ----------
    skeleton : binary np.array; background=0, skeleton=1

    Return
    ------
    vert_df : DataFrame

    """
    save_column = list('xyz')[:skeleton.specs['ndims']]
    save_column.append('nfil')
    vert_df = skeleton.critical_point[skeleton.critical_point.nfil >= 3][save_column]
    return vert_df


def find_edge(skeleton, vert_df, half_edge=True):
    """
    Extract edges. Follow filaments from one vertex to another vertex, and define it as edge.

    Parameters
    ----------
    skeleton  : skel object
    vert_df   : DataFrame of vertices
    half_edge : boolean; 

    Return
    ------
    face_df : DataFrame
    edge_df : DataFrame
    vert_df : DataFrame
    """
    edge_df = pd.DataFrame(dtype='int')

    for i, val in vert_df.iterrows():
        start_cps = np.unique(skeleton.filament[(skeleton.filament.cp1 == i) | (
            skeleton.filament.cp2 == i)][['cp1', 'cp2']])
        for start in start_cps:
            sc = start

            filaments_id = []
            if sc != i:
                # Get the first filament portion
                
                filaments_id.append(skeleton.filament[((skeleton.filament.cp1 == i) & (skeleton.filament.cp2 == sc)) |
                                                      ((skeleton.filament.cp1 == sc)& (skeleton.filament.cp2 == i))].index[0])
                
                previous_sc = i
                previous_previous_sc = previous_sc
                previous_sc = sc
                while skeleton.critical_point.loc[sc]['nfil'] < 3:
                    tmp_sc = np.unique(skeleton.filament[(skeleton.filament.cp1 == previous_sc) | (
                        skeleton.filament.cp2 == previous_sc)][['cp1', 'cp2']])

                    for sc in tmp_sc:

                        if (sc != previous_previous_sc) and (sc != previous_sc):

                            
                            filaments_id.append(skeleton.filament[((skeleton.filament.cp1 == previous_sc) & (skeleton.filament.cp2 == sc)) |
                                                                  ((skeleton.filament.cp1 == sc) & (skeleton.filament.cp2 == previous_sc))].index[0])
                    
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
                         'point_z': pixel_z,
                         'filaments':filaments_id}
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
                if len(set(order_faces[j]).intersection(all_faces[i])) >=2 :
                    order_faces.append(all_faces[i])
                    all_faces.remove(all_faces[i])
                    find = True
                    break
            if find == True:
                break

        if not find:
            break

    edge_df['face'] = -1

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
                logger.warning("there is a problem")
                # print(f)
                # print(tmp)
                # print(tmp_e)
                # print((edge_df.loc[edge]['face'].to_numpy()))
                # print(edge)

        cpt_face += 1

    edge_df.drop(edge_df[edge_df['face']==-1].index, inplace=True)
    edge_df.reset_index(drop=True, inplace=True)
    face_df = pd.DataFrame(index=np.sort(edge_df.face.unique()))
    return face_df, edge_df

def skel_vertices(skel):
    """Return dataframe of the skeleton bifurcation points i.e. the vertices 

    Parameters
    ----------
    skel: skeleton object, has to be breakdowned ('skelconv -breakdown')

    Returns
    -------
    dataframe for the vertices
    """
    df_skel_vertices = skel.critical_point[skel.critical_point.nfil>=3]
    return df_skel_vertices
