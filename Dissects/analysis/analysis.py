
import numpy as np
import scipy as sc
from astropy.convolution import convolve, Tophat2DKernel
import pandas as pd
from segmentation.seg_2D import junction_around_cell

#__all__ = ["cellstats"]

def cellstats(seg, maskfil, im, kernelsize, sigmain, scale):    
    """
    Create a dataframe.

    Parameters
    ----------
    seg : numpy.array
    The segmented image got after applying the segmentation function.

    maskfil : numpy.array
    The mask of filaments.

    im : numpy.array
    The normalised image got after applying the normalise_image function.

    kernelsize : integer

    sigmain : string
    The name of the signal analysed.
    
    scale : integer
    The conversion number -pixel to micrometer- given by import_im
    """
   

    init = np.zeros((len(np.unique(seg)[2:]), 9))
    dataframe = pd.DataFrame(data=init,
                             columns=['CellNbr',
                                      'perimeter_um',
                                      'areaCell_um2',
                                      'meanCell_' + sigmain,
                                      'stdCell_' + sigmain,
                                      'semCell_' + sigmain,
                                      'meanJunc_' + sigmain,
                                      'stdJunc_' + sigmain,
                                      'semJunc_' + sigmain
                                     ])

    for ind, i in enumerate(np.unique(seg)[2:]):
        juncellmaski = junction_around_cell(maskfil, seg, i)
        # enlarge through smoothing 2*KernelSize+1
        juncellmaski_conv = convolve(juncellmaski, Tophat2DKernel(kernelsize))
        juncellmaski_conv[np.where(juncellmaski_conv != 0)] = 1
        dataframe['CellNbr'][ind] = i
        dataframe['perimeter_um'][ind] = len(np.where(juncellmaski == 1)[0])/scale
        dataframe['areaCell_um2'][ind] = len(np.where(seg == i)[0])/scale**2
        dataframe['meanCell_' + sigmain][ind] = np.mean(im[np.where(seg == i)])
        dataframe['stdCell_' + sigmain][ind] = np.std(im[np.where(seg == i)])
        dataframe['semCell_' + sigmain][ind] = sc.stats.sem(im[np.where(seg == i)])
        
        dataframe['meanJunc_' + sigmain][ind] = np.mean(im[np.where(juncellmaski_conv != 0)])
        dataframe['stdJunc_' + sigmain][ind] = np.std(im[np.where(juncellmaski_conv != 0)])
        dataframe['semJunc_' + sigmain][ind] = sc.stats.sem(im[np.where(juncellmaski_conv != 0)])
        

    return dataframe
