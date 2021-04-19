import itertools

import networkx as nx
import numpy as np
import pandas as pd

from io import StringIO
from scipy import ndimage
from sklearn.neighbors import KDTree, BallTree


def generate_segmentation(skeleton_image,
                          free_edges=False,
                          kernel_path='../Dissects/segmentation/3d_pattern.csv'
                          ):
    """
    Generate three dataframe from binary skeleton image which contains informations
    about faces, edges and vertex

    Parameters
    ----------
    skeleton_image : binary np.array; background=0, skeleton=1
    free_edges     : bool; find edge with one side which is not connected
    kernel_path    : str; path to csv file contains list of pattern for vertex detection
    
    Return
    ------
    face_df : DataFrame
    edge_df : DataFrame
    vert_df : DataFrame
    """
    vert_df = find_vertex(skeleton_image, free_edges, kernel_path)
    edge_df = find_edges(skeleton_image, vert_df, half_edge=True)
    face_df, edge_df = find_cell(edge_df)

    return face_df, edge_df, vert_df


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


def find_vertex(skeleton_mask,
                free_edges=False,
                kernel_path='../Dissects/segmentation/3d_pattern.csv'):
    """
    Generate vert_df table from binary image with vertex only
    Advice: make sure to have a skeletonize the output of disperse

    Parameters
    ----------
    skeleton_mask : 
    free_edges    : bool; if True, find vertex extremity
    kernel_path   : str; path to csv file contains list of pattern for vertex detection
    
    Return
    ------
    vert_df: DataFrame of vertex 

    TODO: need to be improve
    """

    kernel = np.array(pd.read_csv(kernel_path, header=None))
    kernel = kernel.reshape((int(kernel.shape[0]/9), 3, 3, 3))

    output_image = np.zeros(skeleton_mask.shape)

    for i in np.arange(len(kernel)):
        out = ndimage.binary_hit_or_miss(skeleton_mask, kernel[i])
        output_image = output_image + out

    if free_edges == True:
        kernel = kernels_extremity()
        for i in np.arange(len(kernel)):
            out = ndimage.binary_hit_or_miss(skeleton_mask, kernel[i])
            output_image = output_image + out

    vert_df = clean_vertex_image(output_image)
    return vert_df


def find_edges(skeleton_image,
               vert_df,
               half_edge=True):
    """
    Find edges with vertex informations. 

    Parameters
    ----------
    skeleton_image : binary np.array; background=0, skeleton=1
    vert_df        : dataframe
    half_edge      : bool

    Return
    ------
    edge_df : Dataframe of edges
    """
    s = ndimage.generate_binary_structure(3, 3)

    # remove vertex + 3x3x3 from initial image
    vertex_img = np.zeros(skeleton_image.shape)
    vertex_img[vert_df['z'].to_numpy(),
                    vert_df['y'].to_numpy(),
                    vert_df['x'].to_numpy()] = 1
    
    vertex_dilation = ndimage.morphology.binary_dilation(vertex_img, structure=s)
    skeleton_without_vertex = img_binary_3d - vertex_dilation

    # Labeled group of isolated pixel which correspond to vertex  
    labeled_array, num_features = ndimage.label(
        skeleton_without_vertex, structure=s)

    # labeled_array
    binary_edges = np.zeros(labeled_array.shape)
    binary_edges = np.where(labeled_array > 0, 1, 0)
    binary_edges = 1.0 * (labeled_array > 0)

    # Initiate edge_df dataframe
    edge_df = pd.DataFrame(index=range(1, num_features+1),
                           columns=['srce', 'trgt'], dtype='int')

    for i, val in vert_df.iterrows():
        img_vert = np.zeros(skeleton_image.shape)
        img_vert[val.z, val.y, val.x] = 1

        img_vert_dilate = ndimage.morphology.binary_dilation(img_vert, structure=s)
        img_corresponding_vertex = img_vert_dilate + binary_edges
        while np.count_nonzero(img_corresponding_vertex == 2) < 2:
            img_vert_dilate = ndimage.morphology.binary_dilation(img_vert_dilate, structure=s)
            img_corresponding_vertex = img_vert_dilate + binary_edges

        edges = labeled_array[np.where(img_corresponding_vertex == 2)]
        for e in np.unique(edges):
            if np.isnan(edge_df.loc[e]['srce']):
                edge_df.loc[e]['srce'] = i
            elif np.isnan(edge_df.loc[e]['trgt']):
                edge_df.loc[e]['trgt'] = i
            else:
                print("problem:", str(i))
                print(edge_df.loc[e])

    # Add pixel from the real shape of edge
    tmp = []
    for e in edge_df.index:
        tmp.append(str(np.where(labeled_array == e)))
    edge_df['points'] = tmp
    edge_df.dropna(axis=0, inplace=True)

    ## Recover small lost junctions
    # Count junction associate to a vertex
    srce_count = np.unique(edge_df.srce, return_counts=True)
    trgt_count = np.unique(edge_df.trgt, return_counts=True)
    res = {}
    for i, v in zip(srce_count[0], srce_count[1]):
        res[i] = res.get(i, 0)+v

    for i, v in zip(trgt_count[0], trgt_count[1]):
        res[i] = res.get(i, 0)+v

    res = pd.DataFrame.from_dict({"idx": res.keys(), "value": res.values()})
    vert_ = res[res.value <= 2]['idx'].to_numpy()
    while len(vert_) > 0:
        X = vert_df[['x', 'y', 'z']].values
        tree = BallTree(X, metric='euclidean')
        dist, ind = tree.query(X[int(vert_[0]-1):int(vert_[0])], 2)

        edge_df.loc[edge_df.index.max()+1] = {'srce': vert_df.index[ind[0][0]],
                                              'trgt': vert_df.index[ind[0][1]]}

        # Count junction associate to a vertex
        srce_count = np.unique(edge_df.srce, return_counts=True)
        trgt_count = np.unique(edge_df.trgt, return_counts=True)
        res = {}
        for i, v in zip(srce_count[0], srce_count[1]):
            res[i] = res.get(i, 0)+v

        for i, v in zip(trgt_count[0], trgt_count[1]):
            res[i] = res.get(i, 0)+v

        res = pd.DataFrame.from_dict(
            {"idx": res.keys(), "value": res.values()})
        vert_ = res[res.value <= 2]['idx'].to_numpy()

    # remove duplicate edges
    edge_df['min'] = np.min(edge_df[['srce', 'trgt']], axis=1)
    edge_df['max'] = np.max(edge_df[['srce', 'trgt']], axis=1)
    edge_df['srce'] = edge_df['min']
    edge_df['trgt'] = edge_df['max']
    edge_df.drop(['min', 'max'], axis=1, inplace=True)
    edge_df.drop_duplicates(inplace=True, subset=['srce', 'trgt'])
    edge_df.reset_index(drop=True, inplace=True)

    if half_edge:
        edge_df = generate_half_edge(edge_df, vert_df)

    return edge_df


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
    new_edge_df = pd.DataFrame(data=[[0, 0, 0]], columns=edge_df.columns)
    for v0, data in vert_df.iterrows():
        va = [edge_df[(edge_df.srce == v0)]['trgt'].to_numpy()]
        va.append(edge_df[(edge_df.trgt == v0)]['srce'].to_numpy())
        va = [item for sublist in va for item in sublist]
        for v in va:
            new_edge_df.loc[
                np.max(new_edge_df.index)+1
            ] = {'srce': v0,
                 'trgt': v,
                 'points': edge_df[
                     ((edge_df.srce == v0) & (edge_df.trgt == v))
                     |
                     (edge_df.trgt == v0) & (edge_df.srce == v)
                 ]['points'].to_numpy()}

            new_edge_df.loc[
                np.max(new_edge_df.index)+1
            ] = {'srce': v,
                 'trgt': v0,
                 'points': edge_df[
                     ((edge_df.srce == v0) & (edge_df.trgt == v))
                     |
                     (edge_df.trgt == v0) & (edge_df.srce == v)
                 ]['points'].to_numpy()}

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
