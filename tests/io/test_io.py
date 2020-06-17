import os
from Dissects.io import load_NDskl
from Dissects.stores import stores_dir


def test_load_NDskl():
    filepath = os.path.join(stores_dir, '3d_images.NDskl.a.NDskl')
    specs, cp, fil, points = load_NDskl(filepath)

    assert specs['ncrit'] == cp.shape[0]
    assert specs['nfil'] == fil.shape[0]
    assert fil['nsamp'].sum() == points.shape[0]


def test_load_image():
    ### verifier le load
    # d'une image 3D
    # d'une image 2D

    # Avec et sans metadata

    return
