import os
from astropy.io import fits
from Dissects.io import (load_NDskl,
                         load_image,
                         save_fits)
from Dissects.stores import stores_dir


def test_load_NDskl():
    filepath = os.path.join(stores_dir, '2d_images.NDskl.a.NDskl')
    cp, fil, points, specs = load_NDskl(filepath)

    assert specs['ncrit'] == cp.shape[0]
    assert specs['nfil'] == fil.shape[0]
    assert fil['nsamp'].sum() == points.shape[0]


def test_load_image():
    # Load 2D image
    filepath = os.path.join(stores_dir, '2d_images.tif')
    image, metadata = load_image(filepath)
    assert len(image.shape) == 2
    assert image.shape[0] == metadata["height"]
    assert image.shape[1] == metadata["width"]

    # Load 3D image
    filepath = os.path.join(stores_dir, '3d_images.tif')
    image, metadata = load_image(filepath)
    assert len(image.shape) == 3
    assert image.shape[0] == metadata["depth"]
    assert image.shape[1] == metadata["height"]
    assert image.shape[2] == metadata["width"]


def test_save_fits():
    filepath = os.path.join(stores_dir, '2d_images.tif')
    image, metadata = load_image(filepath)
    save_fits(image, '2d_fits_image', stores_dir)
    hdul = fits.open(os.path.join(stores_dir, '2d_fits_image.fits'))
    assert hdul[0].header['NAXIS'] == len(image.shape)
    assert hdul[0].header['NAXIS1'] == image.shape[1]
    assert hdul[0].header['NAXIS2'] == image.shape[0]


def test_load_skeleton():
    return


def test_save_skeleton():
    return


def test_save_vtp():
    return
