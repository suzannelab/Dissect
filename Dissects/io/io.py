import os
import re
import warnings
import logging
import exifread

import numpy as np
import pandas as pd

from tifffile import tifffile
from skimage.io import imread, imsave
from tvtk.api import tvtk
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
    cp_df : pd.DataFrame
    fil_df : pd.DataFrame
    fil_point : pd.DataFrame
    specs : Dict

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
        cp_filament_info = {}
        if _check_pattern(_readline(f), '[CRITICAL POINTS]'):
            specs['ncrit'] = int(_readline(f))

            # Bloc of informations for each critical point
            # l1 : type pos, val, pair, boundary
            # l2 : number of filament connected to the CP
            # l3-l_num_fil : filament information
            datas = {}
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
                datas[i] = data
            cp_df = pd.DataFrame.from_dict(datas, orient='index')

        # FILAMENTS
        fils = {}
        cpt_fils = 0
        if _check_pattern(_readline(f), '[FILAMENTS]'):
            specs['nfil'] = int(_readline(f))

            # Bloc of informations for each filament
            # l1 : cp1, cp2, nsamp
            # l2-l... : points informations of filament
            datas = {}
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

                    fils[cpt_fils] = fil
                    cpt_fils += 1

                datas[i] = data
            # Put information in DataFrame
            fil_df = pd.DataFrame.from_dict(datas, orient='index')
            fil_points = pd.DataFrame.from_dict(fils, orient='index')

        # CRITICAL POINT supplementary information
        if _check_pattern(_readline(f), '[CRITICAL POINTS DATA]', optionnal=True):
            ninfo = int(_readline(f))
            crit_columns_name = []
            for i in range(ninfo):
                crit_columns_name.append(_readline(f)[:-1])

            datas = {}
            for i in range(specs['ncrit']):
                data = {}
                line = _readline(f).split()
                for ii in range(ninfo):
                    data[crit_columns_name[ii]] = line[ii]
                datas[i] = data
            cp_supp = pd.DataFrame.from_dict(datas, orient='index')
        # merge cp_df and cp_supp
        cp_df = pd.concat([cp_df, cp_supp], axis=1, sort=False)

        # FILAMENT supplementary information
        if _check_pattern(_readline(f), '[FILAMENTS DATA]', optionnal=True):
            ninfo = int(_readline(f))
            fil_columns_name = []
            for i in range(ninfo):
                fil_columns_name.append(_readline(f)[:-1])

            datas = {}
            for i in range(fil_points.shape[0]):
                data = {}
                line = _readline(f).split()
                for ii in range(ninfo):
                    data[fil_columns_name[ii]] = line[ii]
                datas[i] = data
            fil_supp = pd.DataFrame.from_dict(datas, orient='index')
        # merge cp_df and cp_supp
        fil_points = pd.concat([fil_points, fil_supp], axis=1, sort=False)

    return cp_df, fil_df, fil_points, specs


def load_image(path):
    """ Import a stack of images .TIF in a np.array.

    Parameters
    ----------
    path: str, can be 'tif', 'jpg' ??

    Returns
    -------
    image: np.array
    metadata: dict, contains some meta data of image like pixel ration
    """

    metadata = {}
    with open(path, 'rb') as f:
        tags = exifread.process_file(f)
    p_width = 1 / eval(str(tags['Image XResolution'].values[0]))
    p_height = 1 / eval(str(tags['Image YResolution'].values[0]))
    if p_width == p_height:
        metadata['pixel_ratio'] = p_width
    else:
        warnings.warn("Can't get pixel ratio for now, need to enter manually.")

    image = imread(path)
    if len(image.shape) == 2:
        metadata["height"] = image.shape[0]
        metadata["width"] = image.shape[1]
    else:
        metadata["height"] = image.shape[1]
        metadata["width"] = image.shape[2]
        metadata["depth"] = image.shape[0]

    return image, metadata


def load_skeleton(filestore, data_names=['critical_point', 'filament', 'point']):
    if not os.path.isfile(filestore):
        raise FileNotFoundError("file %s not found" % filestore)
    with pd.HDFStore(filestore) as store:
        data = {name: store[name] for name in data_names if name in store}
    return data


def save_skeleton(skeleton, filename, path=None):
    """ Save skeleton object as HDF5 file

    Parameters
    ----------
    skeleton : skeleton object
    filename: str, name of fits file
    path: str,

    TODO:: Ajouter la sauvegarde des specs.
    """
    warnings.warn("Skeleton object is saved without specs. ")
    if filename[-4:] != '.hf5':
        filename = filename + '.hf5'
    if path is None:
        warnings.warn("Fits file will be saved in the working directory. \
                       Or maybe path is specify in filename...")
        path = os.getcwd()

    filestore = os.path.join(path, filename)

    with pd.HDFStore(filestore) as store:
        store.put('critical_point', skeleton.critical_point)
        store.put('filament', skeleton.filament)
        store.put('point', skeleton.point)


def save_fits(image, filename, path=None):
    """ Convert and save an np.array image into fits file to run Disperse.

    Parameters
    ----------
    image : numpy array
    filename: str, name of fits file
    path: str,
    """
    hdu = fits.PrimaryHDU(image)
    if filename[-5:] != '.fits':
        filename = filename + '.fits'
    if path is None:
        warnings.warn("Fits file will be saved in the working directory. \
                       Or maybe path is specify in filename...")
        path = os.getcwd()

    hdu.writeto(os.path.join(path, filename), overwrite=True)

    logging.info('Saved file: {filename} into {path} directory')


def save_image(image_array, filename, path=None, **kwargs):
    """ Save np.array image into tif file.

    Parameters
    ----------
    image_array : numpy array
    filename: str, name of fits file
    path: str,
    **kwargs: metadata as x_size, y_size, z_size
    """
    if filename[-4:] != '.tif':
        filename = filename + '.tif'
    if path is None:
        warnings.warn("tif file will be saved in the working directory.")
        path = os.getcwd()

    filepath = os.path.join(path, filename)

    x_size = 1
    y_size = 1
    z_size = 1
    if kwargs is None:
        warnings.warn("There is pixel/voxel size defined.")

    else:
        x_size = kwargs["x_size"]
        y_size = kwargs["y_size"]
        try:
            z_size = kwargs["z_size"]
        except:
            pass

    tifffile.imwrite(filepath,
                     image_array.astype('float32'),
                     imagej=True,
                     resolution=(1/x_size, 1/y_size),
                     metadata={'spacing': z_size, 'unit': 'um',
                               'axes': 'XYZCT'})

    logging.info('Saved file: {filename} into {path} directory')


def save_vtp(skeleton, filename, path=None):
    """ Save skeleton as vtp format to see with Paraview software for example.

    Parameters
    ----------
    skeleton:
    filename: str, name of fits file
    path: str,
    TODO:: Add annexe information into the vtp file.
    """

    if filename[-4:] != '.vtp':
        filename = filename + '.vtp'
    if path is None:
        warnings.warn('VTP file will be saved in the working directory.')
        path = os.getcwd()

    filepath = os.path.join(path, filename)

    # get number of critical poin
    nb_points = skeleton.critical_point.shape[0] + skeleton.point.shape[0]

    points = np.zeros((nb_points, skeleton.specs['ndims']))
    verts = np.arange(skeleton.critical_point.shape[0])[:, np.newaxis]
    lines = []

    points[:skeleton.critical_point.shape[0], ] = skeleton.critical_point[
        list('xyz')[:skeleton.specs['ndims']]]
    start = skeleton.critical_point.shape[0]
    points_id = 1
    for i, info in skeleton.filament.iterrows():
        end = start + info.nsamp - 1
        points[start:end, ] = skeleton.point[points_id: points_id +
                                             info.nsamp - 1][list('xyz')[:skeleton.specs['ndims']]]

        line = [info.cp1]
        line.extend(list(range(start, end)))
        line.append(info.cp2)
        lines.append(line)

        points_id += info.nsamp
        start = end + 1

    # vtp need points in 3D, add z plane to 0 for 2D datas.
    if skeleton.specs['ndims'] == 2:
        points = np.concatenate(
            (points, np.zeros((points.shape[0]))[:, np.newaxis]), axis=1)

    vtp_file = tvtk.PolyData(points=points, verts=verts, lines=lines)

    # Add informations

    v = tvtk.XMLPolyDataWriter()
    v.set_input_data(vtp_file)
    v.file_name = filepath
    v.write()

    logging.info('Saved file: {filepath}')
