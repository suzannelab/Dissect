
import warnings
import numpy as np
import pandas as pd

from scipy import stats
from skimage.morphology import binary_dilation
from Dissects.segmentation.seg_2D import junction_around_cell
from Dissects.image import dilation
from scipy import ndimage as ndi
import scipy as sc


def general_analysis(image, mask, normalize=False, noise=None):
    """ Make generale analysis on the image.
    Measure skeleton signal and outside skeleton signal

    Parameters
    ----------
    image : np.array, normalised image got after applying the normalise_image function.
    mask : np.array, mask of filaments with the wanted enlargment
    normalize: bool, default: false, if True, normalize signal with noise
    noise_image: float, noise value to normalize image

    Returns
    -------
    Mean and Std signal of background and skeleton
    """
    if normalize:
        if noise is None:
            warnings.warn(
                "You need to give noise value if you want to normalize the signal.")
            return
        image /= noise
    background_image = (1 - mask) * image
    skeleton_image = mask * image
    mean_background_signal = np.mean(background_image[background_image != 0])
    std_background_signal = np.std(background_image[background_image != 0])

    mean_skeleton_signal = np.mean(skeleton_image[skeleton_image != 0])
    std_skeleton_signal = np.std(skeleton_image[skeleton_image != 0])
    return (mean_background_signal, std_background_signal,
            mean_skeleton_signal, std_skeleton_signal)


def cellstats(image, mask, N, seg, sigmain, scale):
    """
    Create a dataframe.

    Parameters
    ----------
    image : numpy.array
    The normalised image got after applying the normalise_image function.

    mask : numpy.array
    The mask of filaments as the wanted enlargment

    N: integer
    (2*N+1) is the width of the junction

    seg : numpy.array
    The segmented image got after applying the segmentation function.

    sigmain : string
    The name of the signal analysed.

    scale : integer
    The conversion number -pixel to micrometer- given by import_im
    """

    columns_name = ['CellNbr',
                    'perimeter_um',
                    'areaCell_um2',
                    'meanCell_' + sigmain,
                    'stdCell_' + sigmain,
                    'semCell_' + sigmain,
                    'meanJunc_' + sigmain,
                    'stdJunc_' + sigmain,
                    'semJunc_' + sigmain
                    ]
    nb_cells = len(np.unique(seg)[2:])
    init = np.zeros((nb_cells, len(columns_name)))

    dataframe = pd.DataFrame(data=init,
                             columns=columns_name)

    for ind, i in enumerate(np.unique(seg)[2:]):
        dataframe.loc[ind]['CellNbr'] = i

        cell_junction = junction_around_cell(mask, seg, i)
        cell_junction_enlarge = dilation(cell_junction, N)

        image_cell_mask = np.zeros_like(image)
        image_cell_mask[np.where(seg == i)] = 1
        image_cell_reduced = image_cell_mask * \
            (~cell_junction_enlarge.astype(bool)).astype(int)

        image_cell_junction = image[np.where(cell_junction_enlarge != 0)]
        image_cell = image[np.where(image_cell_reduced != 0)]

        dataframe.loc[ind]['perimeter_um'] = len(
            np.where(cell_junction == 1)[0]) / scale
        dataframe.loc[ind]['areaCell_um2'] = len(
            np.where(seg == i)[0]) / scale**2

        # Cytoplasm signal
        dataframe.loc[ind]['meanCell_' + sigmain] = np.mean(image_cell)
        dataframe.loc[ind]['stdCell_' + sigmain] = np.std(image_cell)
        dataframe.loc[ind]['semCell_' + sigmain] = stats.sem(image_cell)

        # Junction signal
        dataframe.loc[ind]['meanJunc_' +
                           sigmain] = np.mean(image_cell_junction)
        dataframe.loc[ind]['stdJunc_' + sigmain] = np.std(image_cell_junction)
        dataframe.loc[ind]['semJunc_' +
                           sigmain] = stats.sem(image_cell_junction)

    return dataframe



def junctions_stats(df_junctions, mask, seg, image, n):
    """"
    Complete the df_junctions DataFrame (created by the 'junctions' function) with signal quantification of the junctions

    Parameters
    ----------
    df_junctions : DataFrame created by the 'junctions' function

    mask : numpy.array
    The mask of filaments as the wanted enlargment

    seg : numpy.array
    The segmented image output of the 'vertices' function

    image: numpy.array
    the image channel you want to realise the quantification upon

    n: integer
    The size (pixel number) of the dilation of the junctions

    Returns
    -------
    DataFrame of the junctions complemented with the direction, mean, standard deviation (std) and standard error of the mean (sem) for each junction
    """

    df_junctions_stats = pd.DataFrame.copy(df_junctions)

    direction=[]
    mean=[]
    std=[]
    sem=[]

    for ind in range (0, df_junctions.shape[0]):

        direction_ind = (np.arctan((df_junctions['sy'][ind]-df_junctions['ty'][ind])
                                        /(df_junctions['sx'][ind]-df_junctions['tx'][ind])))*180/np.pi

        junction_ind = junction_around_cell(mask, seg, df_junctions['Cell1'][ind])*junction_around_cell(mask, seg, df_junctions['Cell2'][ind])

        junctions_ind_dilated = dilation(junction_ind, n)

        mean_ind = np.mean(image[np.where(junctions_ind_dilated != 0)])
        std_ind = np.std(image[np.where(junctions_ind_dilated != 0)])
        sem_ind = sc.stats.sem(image[np.where(junctions_ind_dilated != 0)])

        direction.append(direction_ind)
        mean.append(mean_ind)
        std.append(std_ind)
        sem.append(sem_ind)

    df_junctions_stats['direction'] = direction
    df_junctions_stats['mean'] = mean
    df_junctions_stats['std'] = std 
    df_junctions_stats['sem'] = sem 

    return df_junctions_stats    

