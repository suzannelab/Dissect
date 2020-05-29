# coding: utf-8
import os


def run_disperse(im_dir, im_name, cut, MSC=False):
    """
    Run Disperse on the chosen fits image (in argument).
    Input: image directory, image name (image has to be a fits), cut is the
          threshold (manually here), and MSC default is False change to True if
          you have already run it once and it is stored in .
    Return two files: a fits image and a ascii skeleton
    """

    if MSC:
        os.system('mse ' + im_dir + im_name + ' -outDir ' + im_dir + ' -loadMSC ' +
                  im_dir + im_name + '.MSC' + ' -cut ' + cut + ' -periodicity 0 -upSkl')
    else:
        os.system('mse ' + im_dir + im_name + ' -outDir ' + im_dir +
                  ' -cut ' + cut + ' -periodicity 0 -upSkl')

    os.system('skelconv ' + im_dir + im_name + '_c' + cut +
              '.up.NDskl -outDir ' + im_dir + ' -toFITS')
    os.system('skelconv ' + im_dir + im_name + '_c' + cut +
              '.up.NDskl -outDir ' + im_dir + ' -to NDskl_ascii')

    return
