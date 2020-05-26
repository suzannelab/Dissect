import numpy as np
from sklearn import manifold
from skimage import morphology


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
    training_data : array, training vector, subset of datas
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
            img_binary[round(points[1][i] + (ncols / 2)).astype(int),
                       round(points[0][i] + (nrows / 2)).astype(int)] = 1

    else:
        df['x_b'] = 0
        df['y_b'] = 0
        for i in range(point_transformed.shape[1]):
            img_binary[round(points[1][i] + (ncols / 2)).astype(int),
                       round(points[0][i] + (nrows / 2)).astype(int)] = 1
            df.loc[i, 'x_b'] = round(points[1][i] + (ncols / 2)).astype(int)
            df.loc[i, 'y_b'] = round(points[0][i] + (nrows / 2)).astype(int)

    # voir ce qui peut etre améliorer pour fermer les cellules
    img_binary = morphology.binary_dilation(img_binary)
    img_binary = morphology.binary_dilation(img_binary)
    img_binary = morphology.skeletonize(img_binary)

    return img_binary
