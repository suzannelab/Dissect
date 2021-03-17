import networkx as nx
import numpy as np
import pandas as pd
import scipy as sci
import itertools

from sklearn import manifold
from skimage import morphology
from .seg_2D import generate_mesh
from scipy.ndimage.morphology import binary_dilation


def find_vertex(skeleton_mask,
                free_edges=False,
                kernel_path='../Dissects/segmentation/3d_pattern.csv'):
    """
    free_edges : if True, find vertex extremity
    warning :  make sure to have a skeletonize the output of disperse
    """

    # Need to be improve
    kernel = np.array(pd.read_csv(kernel_path, header=None))
    kernel = kernel.reshape((int(kernel.shape[0]/9), 3, 3, 3))

    output_image = np.zeros(skeleton_mask.shape)

    for i in np.arange(len(kernel)):
        out = sci.ndimage.binary_hit_or_miss(skeleton_mask, kernel[i])
        output_image = output_image + out

    if free_edges == True:
        kernel = kernels_extremity()
        for i in np.arange(len(kernel)):
            out = sci.ndimage.binary_hit_or_miss(skeleton_mask, kernel[i])
            output_image = output_image + out

    return output_image


def clean_vertex(vertex_image):
    s = sci.ndimage.generate_binary_structure(3, 3)
    labeled_array, num_features = sci.ndimage.label(vertex_image, structure=s)
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


def find_edges(img_binary_3d,
               output_vertex,
               vert_df,
               half_edge=True):
    # remove vertex +3x3x3 from initial image
    img_binary_3d_without_vertex = img_binary_3d.copy()
    for i, p in vert_df.iterrows():
        for z_ in range(int(p.z)-1, int(p.z)+2):
            for y_ in range(int(p.y)-1, int(p.y)+2):
                for x_ in range(int(p.x)-1, int(p.x)+2):
                    try:
                        img_binary_3d_without_vertex[z_][y_][x_] = 0
                    except:
                        pass
    s = sci.ndimage.generate_binary_structure(3, 3)
    labeled_array, num_features = sci.ndimage.label(
        img_binary_3d_without_vertex, structure=s)

    # labeled_array

    binary_edges = np.zeros(labeled_array.shape)
    binary_edges = np.where(labeled_array > 0, 1, 0)

    # clean output vertex
    output_vertex = np.zeros(output_vertex.shape)
    for i, val in vert_df.iterrows():
        output_vertex[val.z, val.y, val.x] = 1

    # Initiate edge_df dataframe
    edge_df = pd.DataFrame(index=range(1, num_features+1),
                           columns=['srce', 'trgt'], dtype='int')

    for i, val in vert_df.iterrows():
        img_vert = np.zeros(output_vertex.shape)

        img_vert[val.z, val.y, val.x] = 1

        s = sci.ndimage.generate_binary_structure(3, 3)

        img_vert_dilate = binary_dilation(img_vert, structure=s)
        img_corresponding_vertex = img_vert_dilate + binary_edges
        while np.count_nonzero(img_corresponding_vertex == 2) < 2:
            img_vert_dilate = binary_dilation(img_vert_dilate, structure=s)
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

    tmp = []
    for e in edge_df.index:
        tmp.append(str(np.where(labeled_array == e)))
    edge_df['points'] = tmp

    edge_df.dropna(axis=0, inplace=True)

    # recupère les petites jonctions perdus

    from sklearn.neighbors import KDTree, BallTree
    from io import StringIO

    # Compte le nombre de jonction associé à un vertex
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

        # Compte le nombre de jonction associé à un vertex
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

    # remove doublon
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

    # Pour chaque face
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

    return edge_df
