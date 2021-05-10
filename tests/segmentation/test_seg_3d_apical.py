import os

import numpy as np
import pandas as pd

from Dissects.stores import stores_dir
from Dissects.segmentation import seg_dir
from Dissects.io import load_image
from Dissects.segmentation.seg_3D_apical import (generate_segmentation,
                                                 find_vertex,
                                                 clean_vertex_image)


def test_generate_segmentation():
    filepath = os.path.join(stores_dir, '3d_skeleton.tif')
    img_skeleton, metadata = load_image(filepath)

    face_df, edge_df, vert_df = generate_segmentation(img_skeleton,
                                                      kernel_path=os.path.join(
                                                          seg_dir, '3d_pattern.csv'),
                                                      clean=True)
    assert vert_df.shape[0] == 198
    assert vert_df.shape[1] == 3
    assert edge_df.shape[0] == 594
    assert face_df.shape[0] == 101


# def test_find_vertex():
#     filepath = os.path.join(stores_dir, '3d_skeleton.tif')
#     img_skeleton, metadata = load_image(filepath)
#     vert_df = find_vertex(img_skeleton,
#                           kernel_path=os.path.join(seg_dir, '3d_pattern.csv'),
#                           clean=False)
#     assert vert_df.shape[0] == 287
#     assert vert_df.shape[1] == 3


# def test_clean_vertex():
#     filepath = os.path.join(stores_dir, '3d_skeleton.tif')
#     img_skeleton, metadata = load_image(filepath)

#     vert_df = find_vertex(img_skeleton,
#                           kernel_path=os.path.join(seg_dir, '3d_pattern.csv'),
#                           clean=True)
#     assert vert_df.shape[0] == 198
#     assert vert_df.shape[1] == 3


# def test_find_edges():
#     filepath = os.path.join(stores_dir, '3d_skeleton.tif')
#     img_skeleton, metadata = load_image(filepath)

#     vert_df = find_vertex(img_skeleton,
#                           kernel_path=os.path.join(seg_dir, '3d_pattern.csv'),
#                           clean=True)

def test_find_cell():
	return

	
def test_generate_half_edge():
	return

