import os
from astropy.io import fits
from Dissects.io import (load_NDskl,
                         load_image,
                         save_fits,
                         load_skeleton,
                         save_skeleton,
                         save_vtp)
from Dissects.geometry import Skeleton
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
    filepath = os.path.join(stores_dir, '2d_skeleton.hf5')
    data = load_skeleton(filepath)
    assert 'critical_point' in data.keys()
    assert 'filament' in data.keys()
    assert 'point' in data.keys()


def test_save_skeleton():
    filepath = os.path.join(stores_dir, '2d_skeleton.hf5')
    data = load_skeleton(filepath)
    skel = Skeleton(data['critical_point'],
                    data['filament'],
                    data['point'])
    save_skeleton(skel, 'test_save_skel.hf5', stores_dir)

    data2 = load_skeleton(os.path.join(stores_dir, 'test_save_skel.hf5'))
    assert data['critical_point'].shape == data['critical_point'].shape
    assert data['filament'].shape == data['filament'].shape
    assert data['point'].shape == data['point'].shape
    os.remove(os.path.join(stores_dir, 'test_save_skel.hf5'))


def test_save_vtp():
    filepath = os.path.join(stores_dir, '2d_skeleton.hf5')
    data = load_skeleton(filepath)
    skel = Skeleton(data['critical_point'],
                    data['filament'],
                    data['point'])
    skel.specs['ndims'] = 2
    save_vtp(skel, 'test_save_vtp.vtp', stores_dir)
    assert os.path.isfile(os.path.join(stores_dir, 'test_save_vtp.vtp'))

    os.remove(os.path.join(stores_dir, 'test_save_vtp.vtp'))
    
