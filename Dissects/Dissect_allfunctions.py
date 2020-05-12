# coding: utf-8
import os
import sys
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import find_peaks
from astropy.convolution import convolve,Tophat2DKernel,Box2DKernel
from PIL import Image
from pathlib import Path
import pandas as pd

import glob
from joblib import Parallel, delayed
import multiprocessing
import pathlib

import skimage
from skimage import filters as filters
from skimage.morphology import disk
from skimage.filters.rank import threshold
from skimage.filters.rank import mean
from skimage.filters.rank import enhance_contrast
from skimage.io.collection import ImageCollection,MultiImage
from skimage.io import imread
from matplotlib.colors import LogNorm
from scipy.ndimage.filters import gaussian_filter


from skimage.filters import sobel
from scipy import ndimage as ndi
from skimage.measure import label
from skimage.morphology import watershed   
from skimage.morphology import remove_small_objects
import time

import scipy.stats as st

import seaborn as sns
from scipy.stats import pearsonr

#import pingouin as pg




def otsufilter(img):
    '''
    This function applies an Otsu filter, it takes in arguments a matrix and return a mask where 
    the background is set to 0 and the foreground to 1. Change nbins with the type of images (8bits = 256, 16bits = 65536
    '''
    
    val = filters.threshold_otsu(img, nbins = 65536)
    mask = img < val
    mask = np.invert(mask)
    mask = sc.ndimage.binary_fill_holes(mask)
    return mask











def stack_importation(chemin):
    '''
    This function imports a stack of images .TIF in a 3D matrix, it takes in arguments the path of the folder.
    '''
    
    img = MultiImage(chemin)
    size = len(img)
    if (size != 1) :
        shape = img[0].shape
        stack = np.zeros((shape[0],shape[1],size))
        for i in range(size):
            stack[:,:,i] = img[i]
    if (size == 1):
        shape = img[0].shape
        stack = np.zeros((shape[1],shape[2],shape[0]))
        for i in range(shape[0]):
            stack[:,:,i] = img[0][i,:,:]
        shape=stack[:,:,0].shape
    return stack, size, shape


def proj_aroundMAX(matrix,shape,number):
    '''
    This function makes a maximum intensity projection of a 3D matrix, it takes in arguments 
	-the 3D matrix
	-the shape of the 3D matrix in a tuple 
	-and the number of points around the max to average. The mean will be calculated with 2n+1 points (middle point = the max)
    '''
    projection = np.zeros(shape)
	for x in range(shape[0]):
        for y in range(shape[1]):
            ind_max = np.argmax(matrix[x, y, :])
            inf = max(ind_max-number,0)
            sup = min(ind_max+(number+1),matrix.shape[2])
            projection[x, y] = np.mean(matrix[x, y, inf:sup])
    return projection



def tif2fits(base,chemin,stack=True,N=None):
    '''
    Function to convert .tif to .fits  
	Use proj_aroundMAX to define the width of projection
    '''
    if stack:
        im3d,im3d_size,im3d_shape = stack_importation(chemin)
        if N == None:
            return 'Error : You said TIF image is a stack, need to provide keyword N'
        else:
            im = proj_aroundMAX(im3d,im3d_shape,N)
    else:
        im = Image.open(chemin)
    hdu = fits.PrimaryHDU(im)
    hdu.writeto(base,overwrite=True)
    return



def RunDisperse2D(ImDir,ImName,Threshold,MSC=False):
    '''
    This function runs Disperse on the chosen image (in argument). Choose the threshold manually here (not automated YET), 
    and MSC=False for ce 1rst time you run it. MSC=True after.
    '''
    if MSC:
        os.system('mse '+ImDir+ImName+' -outDir '+ImDir+' -loadMSC '+ImDir+ImName+'.MSC'+' -cut '+Threshold+' -periodicity 0 -upSkl')
    else:
        os.system('mse '+ImDir+ImName+' -outDir '+ImDir+' -cut '+Threshold+' -periodicity 0 -upSkl')                
    os.system('skelconv '+ImDir+ImName+'_c'+Threshold+'.up.NDskl -outDir '+ImDir+' -toFITS')
    os.system('skelconv '+ImDir+ImName+'_c'+Threshold+'.up.NDskl -outDir '+ImDir+' -to NDskl_ascii')    
    return




def clean_skeleton(skeleton, save=True):
    """
    Function that cleans skeleton.
    Means that all isolated filament or critical point will be removed.
    """
    keep_going = True

    while keep_going:
        keep_going = False

        # Mark critical point connected to 1 filament or less
        cp_to_keep = []
        for c in skeleton.crit:
            if c.nfil < 2:
                cp_to_keep.append(False)
            else:
                cp_to_keep.append(True)
        # Remove critical point connected to 1 filament or less
        skeleton.crit[:] = np.array(skeleton.crit)[cp_to_keep]

        # Mark filament connected to only one critical point
        fil_to_keep = []
        for f in skeleton.fil:
            if f.cp1 not in skeleton.crit:
                fil_to_keep.append(False)
                for i in range(len(skeleton.crit)):
                    if f.cp2 == skeleton.crit[i]:
                        keep_going = True
                        skeleton.crit[i].filId.remove(f)
                        skeleton.crit[i].destCritId.remove(f.cp1)
                        break

            elif f.cp2 not in skeleton.crit:
                fil_to_keep.append(False)
                for i in range(len(skeleton.crit)):
                    if f.cp1 == skeleton.crit[i]:
                        keep_going = True
                        skeleton.crit[i].filId.remove(f)
                        skeleton.crit[i].destCritId.remove(f.cp2)
                        break

            else:
                fil_to_keep.append(True)

        # Remove filament
        skeleton.fil[:] = np.array(skeleton.fil)[fil_to_keep]

    if skeleton.isValid():
        if save:
            skeleton.write_vtp(os.path.join(skeleton._filename, "_removefil.vtp"))
    else:
        raise nameError('skeleton not valid')





def FilMask_int(skeleton,im):
#cette fonction permet d'assigner une position entière aux points des segments. 
#Si la coordonnée est plus grande que la taille de l'image, elle est ramenée au bord. 


    mask = np.zeros_like(im)
    for i in range(skeleton.nfil):
        for j in range(len(skeleton.fil[i].points)):
            ii = skeleton.fil[i].points[j,1].astype(int)
            if ii < 0:
                ii = 0
            if ii >= mask.shape[0]:
                ii = mask.shape[0]-1        
            jj = skeleton.fil[i].points[j,0].astype(int)
            if jj < 0:
                jj = 0
            if jj >= mask.shape[1]:
                jj = mask.shape[1]-1            
            mask[ii,jj] = 1
    return mask




def NormaliseImage(im,KernelSize):
    smoothIm = convolve(im,Tophat2DKernel(KernelSize))
    otsu = otsufilter(smoothIm)
    background = np.mean(im[np.where(~otsu)])
    normIm = im / background   
    return normIm





def segmentation(InvMaskFil,min_area):
    edges = sobel(InvMaskFil)
    markers = np.zeros_like(InvMaskFil)
    markers[InvMaskFil == 0] = 1
    markers[InvMaskFil > 0] = 2

    segmentation = watershed(edges, markers)  
    segmentation1,_ = ndi.label(segmentation == 2)
    seg = remove_small_objects(segmentation1, min_area)

    return seg








def JuncCell(seg,MaskFil,i):
#trouve les jonctions autour de la cellule i 
    segmentationi = np.zeros_like(seg)
    #for each cell get contour pixels
    segmentationi[np.where(seg == i)] = 1

    #Box smooth around unique cell + multiply by MaskFil to have pixel filaments
    kernel = Box2DKernel(2)
    JuncCelli = (convolve(segmentationi,kernel)*MaskFil).astype(bool).astype(int)     
    return JuncCelli







def CellStatsMain(seg,MaskFil,Im,KernelSize,sigMain):
#prend en argument: la segmentation, le masque des filaments, l'image originale, taille d'élargissement des filaments, 
#et le nom du signal avec lequel la segmentation est faite
#créé un tableau panda avec pour le signal de l'image originale donné et pour chaque cellule les stats int et ext
#(moyenne, std, sem, area int, perimetre)

    init = np.zeros((len(np.unique(seg)[2:]),9))
    Dataframe = pd.DataFrame(data=init,columns=['CellNbr','meanCell_'+sigMain,'stdCell_'+sigMain,'semCell_'+sigMain,'areaCell','meanJunc_'+sigMain,'stdJunc_'+sigMain,'semJunc_'+sigMain,'perimeter'])

    for ind,i in enumerate(np.unique(seg)[2:]):
        JuncCellMaski = JuncCell(seg,MaskFil,i)    
        # enlarge through smoothing 2*KernelSize+1
        JuncCellMaski_conv = convolve(JuncCellMaski,Tophat2DKernel(KernelSize))
        JuncCellMaski_conv[np.where(JuncCellMaski_conv != 0)] = 1   
        ### multiply this mask of filaments around cell i with pixels from Im
        #JuncCell_i=JuncCellMaski_conv*Im
        #compute mean and std of bcat in cells and actine in filaments
        Dataframe['CellNbr'][ind] = i
        Dataframe['meanCell_'+sigMain][ind] = np.mean(Im[np.where(seg == i)])
        Dataframe['stdCell_'+sigMain][ind] = np.std(Im[np.where(seg == i)])
        Dataframe['semCell_'+sigMain][ind] = sc.stats.sem(Im[np.where(seg == i)])
        Dataframe['areaCell'][ind] =  len(np.where(seg == i)[0])
        Dataframe['meanJunc_'+sigMain][ind] = np.mean(Im[np.where(JuncCellMaski_conv != 0)])
        Dataframe['stdJunc_'+sigMain][ind] = np.std(Im[np.where(JuncCellMaski_conv != 0)])
        Dataframe['semJunc_'+sigMain][ind] = sc.stats.sem(Im[np.where(JuncCellMaski_conv != 0)])
        Dataframe['perimeter'][ind] = len(np.where(JuncCellMaski == 1)[0])

    return Dataframe




