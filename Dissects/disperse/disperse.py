# coding: utf-8
import os
import sys

def RunDisperse2D(imDir, imName, cut, MSC=False):
    '''
    This function runs Disperse on the chosen fits image (in argument). 
        Input: image directory, image name (image has to be a fits), cut is the 
        threshold (manually here), and MSC default is False change to True if you have 
        already run it once and it is stored in .
        Returns two files: a fits image and a ascii skeleton
    '''

    if MSC:
        os.system('mse ' + imDir + imName + ' -outDir ' + imDir + ' -loadMSC ' +
                  imDir + imName + '.MSC' + ' -cut ' + cut + ' -periodicity 0 -upSkl')
    else:
        os.system('mse ' + imDir + imName + ' -outDir ' + imDir +
                  ' -cut ' + cut + ' -periodicity 0 -upSkl')

    os.system('skelconv ' + imDir + imName + '_c' + cut +
              '.up.NDskl -outDir ' + imDir + ' -toFITS')
    os.system('skelconv ' + imDir + imName + '_c' + cut +
              '.up.NDskl -outDir ' + imDir + ' -to NDskl_ascii')

    return
