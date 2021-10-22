import numpy as np
import pandas as pd

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



