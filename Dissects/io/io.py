import os
import re
import warnings
import logging
import exifread

import numpy as np
import pandas as pd

from skimage.io import imread
from astropy.io import fits


def _readline(f):
    for line in f:
        # Skip comment line
        if line.find('#') != 0:
            break
    return line


def _check_pattern(line, pattern, optionnal=False):
    if line.find(pattern) != 0:
        if not(optionnal):
            raise ValueError('Wrong format, missing {}'.format(pattern))
    else:
        return line


def load_NDskl(filename):
    """ Read NDskl file and generate two dataframe which
    contains critical point and filament

    Parameters
    ----------
    filename: str, Path to the file

    Returns
    -------
    specs : Dict
    cp_df : pd.DataFrame
    fil_df : pd.DataFrame
    fil_point : pd.DataFrame

    """
    specs = {}

    with open(filename, 'r') as f:

        # 2D or 3D images
        if _check_pattern(_readline(f), 'ANDSKEL'):
            specs['ndims'] = int(_readline(f))

        # Size of the image
        line = _check_pattern(_readline(f), 'BBOX', optionnal=True)
        if line:
            bbox, bbox_delta = re.findall('\[.*?\]', line)
            specs['bbox'] = np.asfarray(bbox[1:-1].split(','))
            specs['bbox_delta'] = np.asfarray(bbox_delta[1:-1].split(','))

        # CRITICAL POINTS
        cp_column = (list('xyz')[:specs['ndims']] +
                     ['type', 'val', 'pair', 'boundary', 'nfil'])
        cp_df = pd.DataFrame(columns=cp_column)
        cp_filament_info = {}
        if _check_pattern(_readline(f), '[CRITICAL POINTS]'):
            specs['ncrit'] = int(_readline(f))

            # Bloc of informations for each critical point
            # l1 : type pos, val, pair, boundary
            # l2 : number of filament connected to the CP
            # l3-l_num_fil : filament information
            for i in range(specs['ncrit']):
                data = {}
                # l1
                line1 = _readline(f).split()
                data['type'] = int(line1[0])
                for n in range(specs['ndims']):
                    data[list('xyz')[n]] = float(line1[1 + n])
                data['val'] = float(line1[1 + specs['ndims']])
                data['pair'] = float(line1[2 + specs['ndims']])
                data['boundary'] = float(line1[3 + specs['ndims']])
                # l2
                data['nfil'] = int(_readline(f))
                # l3-l_num_fil
                for _ in range(data['nfil']):
                    line = _readline(f).split()
                    cp_filament_info[i] = {'destcritid': int(line[0]),
                                           'fillId': int(line[1])}
                # Put information in DataFrame
                cp_df = cp_df.append(data, ignore_index=True)

        # FILAMENTS
        fil_column = ['cp1', 'cp2', 'nsamp']
        fil_df = pd.DataFrame(columns=fil_column)
        fil_points = pd.DataFrame(columns=list('xyz')[:specs['ndims']])
        if _check_pattern(_readline(f), '[FILAMENTS]'):
            specs['nfil'] = int(_readline(f))

            # Bloc of informations for each filament
            # l1 : cp1, cp2, nsamp
            # l2-l... : points informations of filament
            for i in range(specs['nfil']):
                data = {}
                # l1
                line1 = _readline(f).split()
                data['cp1'] = int(line1[0])
                data['cp2'] = int(line1[1])
                data['nsamp'] = int(line1[2])
                for _ in range(data['nsamp']):
                    line = _readline(f).split()
                    fil = {}
                    for n in range(specs['ndims']):
                        fil[list('xyz')[n]] = float(line[n])
                        fil['filament'] = i

                    fil_points = fil_points.append(fil, ignore_index=True)
                # Put information in DataFrame
                fil_df = fil_df.append(data, ignore_index=True)

        # CRITICAL POINT supplementary information
        if _check_pattern(_readline(f), '[CRITICAL POINTS DATA]', optionnal=True):
            ninfo = int(_readline(f))
            crit = []
            for i in range(ninfo):
                crit.append(_readline(f)[:-1])
            cp_supp = pd.DataFrame(columns=crit)
            for i in range(specs['ncrit']):
                data = {}
                line = _readline(f).split()
                for ii in range(ninfo):
                    data[crit[ii]] = line[ii]
                cp_supp = cp_supp.append(data, ignore_index=True)
        # merge cp_df and cp_supp
        cp_df = pd.concat([cp_df, cp_supp], axis=1, sort=False)

        # FILAMENT supplementary information
        if _check_pattern(_readline(f), '[FILAMENTS DATA]', optionnal=True):
            ninfo = int(_readline(f))
            fil = []
            for i in range(ninfo):
                fil.append(_readline(f)[:-1])
            fil_supp = pd.DataFrame(columns=fil)
            for i in range(fil_points.shape[0]):
                data = {}
                line = _readline(f).split()
                for ii in range(ninfo):
                    data[crit[ii]] = line[ii]
                fil_supp = fil_supp.append(data, ignore_index=True)
        # merge cp_df and cp_supp
        fil_points = pd.concat([fil_points, fil_supp], axis=1, sort=False)

    return specs, cp_df, fil_df, fil_points


def load_image(path):
    """ Import a stack of images .TIF in a np.array.

    Parameters
    ----------
    path: str, can be 'tif', 'jpg' ??

    Returns
    -------
    metadata: dict, contains some meta data of image like pixel ration
    image: np.array
    """

    metadata = {}
    with open(path, 'rb') as f:
        tags = exifread.process_file(f)
    p_width = 1 / eval(str(tags['Image XResolution'].values[0]))
    p_height = 1 / eval(str(tags['Image YResolution'].values[0]))
    if p_width == p_height:
        metadata['pixel_ratio'] = p_width
    else:
        warnings.warn("The ratio is not the same in x and y axis.")

    image = imread(path)
    return image, metadata


def save_fits(image, filename, path=None):
    """ Convert and save an np.array image into fits file.

    Parameters
    ----------
    image : numpy array
    filename: str, name of fits file
    path: str,
    """
    hdu = fits.PrimaryHDU(image)
    filename = filename + '.fits'
    if path is None:
        warnings.warn("Fits file will be saved in the working directory.")
        path = os.getcwd()

    hdu.writeto(os.path.join(path, filename), overwrite=True)

    logging.info('Saved file: {filename} into {path} directory')


def save_vtp(skeleton):
    return
