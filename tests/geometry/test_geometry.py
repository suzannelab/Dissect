import os

import numpy as np

from Dissects.io import load_skeleton
from Dissects.stores import stores_dir
from Dissects.geometry import Skeleton


def test_skeleton():
    filepath = os.path.join(stores_dir, '2d_skeleton.hf5')
    data = load_skeleton(filepath)
    skel = Skeleton(data['critical_point'],
                    data['filament'],
                    data['point'],)


def test_remove_lonely_cp():
    filepath = os.path.join(stores_dir, '2d_skeleton.hf5')
    data = load_skeleton(filepath)
    skel = Skeleton(data['critical_point'],
                    data['filament'],
                    data['point'])

    skel.remove_lonely_cp()

    assert skel.critical_point.shape[0] == 3365


def test_remove_free_filament():
    filepath = os.path.join(stores_dir, '2d_skeleton.hf5')
    data = load_skeleton(filepath)
    skel = Skeleton(data['critical_point'],
                    data['filament'],
                    data['point'])

    skel.remove_free_filament()

    assert skel.critical_point.shape[0] == 2827
    assert skel.filament.shape[0] == 2824
    assert skel.point.shape[0] == 13969


def test_binary_image():
    filepath = os.path.join(stores_dir, '2d_skeleton.hf5')
    data = load_skeleton(filepath)
    skel = Skeleton(data['critical_point'],
                    data['filament'],
                    data['point'])
    skel.specs['bbox_delta'] = np.array([576, 337])
    skel.specs['ndims'] = 2

    image = skel.create_binary_image()
    assert image.shape == (337, 576)
    assert np.where
