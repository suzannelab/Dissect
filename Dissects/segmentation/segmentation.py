import numpy as np
import pandas as pd

from scipy import ndimage

from ..geometry import Skeleton

class Segmentation:

    def __init__(self):
        self.face_df = pd.DataFrame()
        self.edge_df = pd.DataFrame()
        self.vert_df = pd.DataFrame()
        self.point_df = pd.DataFrame()
        self.specs = dict()


    def __init__(self, skeleton):
        self.skeleton = skeleton
        self.face_df = pd.DataFrame()
        self.edge_df = pd.DataFrame()
        self.vert_df = pd.DataFrame()
        self.point_df = pd.DataFrame()
        self.specs = dict()


    def __init__(self, skeleton, specs):
        self.skeleton = skeleton
        self.face_df = pd.DataFrame()
        self.edge_df = pd.DataFrame()
        self.vert_df = pd.DataFrame()
        self.point_df = pd.DataFrame()
        self.specs = specs


    def image_vertex(self):
        return

    def image_junction(self):
        return

    def image_face(self):
        return

    def compute_normal(self):
        return

    def update_geom(self):
        return