import numpy as np
import pandas as pd
import scipy as sci
import itertools

from sklearn import manifold
from skimage import morphology
from .seg_2D import generate_mesh


def flatten_tissu(datas,
                  training_data=None,
                  n_neighbors=50,
                  n_components=2,
                  n_jobs=1):
    """Isomap tissu flatten

    Use Isomap principle to reduce dimension and pass from a 3D
    problem through a 2D.

    Parameters
    ----------
    datas : array, complete data set
    training_data : array, training vector, subset of datas, usually correspond to cp from skeleton
    n_neighbors : int, number of neighbors to consider for each point
    n_components : int, number of coordinates for the manifold
    n_jobs : int, number of parallel jobs to run

    Returns
    -------
    point_ : array, new position of datas in 2D
    """
    if training_data is None:
        training_data = datas

    embedding = manifold.Isomap(n_neighbors=n_neighbors,
                                n_components=n_components,
                                n_jobs=n_jobs)

    # Compute the embedding vectors for data X
    embedding.fit(training_data)
    # Apply on big data set
    point_ = embedding.transform(datas).T

    return point_


def binary_flatten_tissu(point_transformed, df=None):
    """ Create binary image from a flatten tissu
    Parameters
    ----------
    point_transformed
    df :
    Returns
    -------
    img_binary : array, binary image
    """

    # Arrondi à la centaine supérieure
    nrows = int(max(abs(min(point_transformed[0])), max(
        point_transformed[0])) * 2 // 100 + 1) * 100
    ncols = int(max(abs(min(point_transformed[1])), max(
        point_transformed[1])) * 2 // 100 + 1) * 100
    img_binary = np.zeros([ncols, nrows], dtype=np.uint8)
    points = np.array([point_transformed[0], point_transformed[1]])

    if df is None:
        for i in range(point_transformed.shape[1]):
            img_binary[round(points[1][i] + (ncols / 2)),
                       round(points[0][i] + (nrows / 2))] = 1

    else:
        df['x_b'] = 0
        df['y_b'] = 0
        for i in range(point_transformed.shape[1]):
            img_binary[round(points[1][i] + (ncols / 2)),
                       round(points[0][i] + (nrows / 2))] = 1
            df.loc[i, 'x_b'] = round(points[1][i] + (ncols / 2))
            df.loc[i, 'y_b'] = round(points[0][i] + (nrows / 2))

    # voir ce qui peut etre améliorer pour fermer les cellules
    img_binary = morphology.binary_dilation(img_binary)
    img_binary = morphology.binary_dilation(img_binary)
    img_binary = morphology.skeletonize(img_binary)

    return img_binary.astype(int)


def generate_mesh_3D(mask, df_convert):

    face_df, edge_df, vert_df = generate_mesh(mask)

    vert_df_3d = pd.DataFrame()
    for _, v in vert_df.iterrows():

        df_ = df_convert[df_convert['x_b'] == v.x]
        if not df_.empty:
            df_ = df_[df_['y_b'] == v.y]
            if not df_.empty:
                vert_df_3d = vert_df_3d.append({'x': df_.x.to_numpy()[0],
                                                'y': df_.y.to_numpy()[0],
                                                'z': df_.z.to_numpy()[0]},
                                               ignore_index=True)
            else:
                df_ = df_convert[df_convert['x_b'] == v.x]
                df_ = df_.iloc[np.argmin(np.abs(df_.y_b - v.y))]
                vert_df_3d = vert_df_3d.append({'x': df_.x,
                                                'y': df_.y,
                                                'z': df_.z},
                                               ignore_index=True)

    return face_df, edge_df, vert_df_3d



def find_vertex(mask, free_edges=False):
    """
    free_edges : if True, find vertex extremity
    """
    # make sure to have a skeleton
    skeleton_mask = morphology.skeletonize(mask)

    kernel = kernels_3d()
    output_image = np.zeros(skeleton_mask.shape)

    for i in np.arange(len(kernel)):
        out = sci.ndimage.binary_hit_or_miss(skeleton_mask, kernel[i] )
        output_image = output_image + out

    if free_edges==True:
        kernel = kernels_extremity()
        for i in np.arange(len(kernel)):
            out = sci.ndimage.binary_hit_or_miss(skeleton_mask, kernel[i] )
            output_image = output_image + out

    return output_image


def kernels_3d():
    # Need to write some kernels
    # Idealy if it can learn it could be very nice...
    # Especially for 3d kernel...
    kernels = np.array(
                       np.array([[[0,0,0],
                                  [0,1,0],
                                  [0,0,0]],
                                 [[0,1,0],
                                  [1,1,1],
                                  [0,1,0]],
                                 [[0,0,0],
                                  [0,1,0],
                                  [0,0,0]]]),

                       np.array([[[0,0,0],
                                  [0,1,0],
                                  [0,0,0]],
                                 [[0,1,0],
                                  [1,1,0],
                                  [0,1,0]],
                                 [[0,0,1],
                                  [0,0,0],
                                  [0,0,0]]]),

                       )



    return kernels

def kernels_extremity():
    kernels = np.array(
                       np.array([[[0,0,0],
                                  [0,1,0],
                                  [0,0,0]],
                                 [[0,0,0],
                                  [0,1,0],
                                  [0,0,0]],
                                 [[0,0,0],
                                  [0,0,0],
                                  [0,0,0]]]),

                       np.array([[[0,0,0],
                                  [0,0,0],
                                  [0,0,0]],
                                 [[0,0,0],
                                  [1,1,0],
                                  [0,0,0]],
                                 [[0,0,0],
                                  [0,0,0],
                                  [0,0,0]]]),

                       )

    return kernels