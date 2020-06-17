import numpy as np
import pandas as pd

import warnings


class Skeleton():

    # def __init__(self):
    #     """ Initialisation of an empty skeleton

    #     """
    #     self.critical_point = pd.DataFrame()
    #     self.filament = pd.DataFrame()
    #     self.point = pd.DataFrame()
    #     self.specs = {}

    def __init__(self, cp_df, fil_df, point_df, specs={}):
        """ Create an skeleton object

        Parameters
        ----------
        cp_df: pd.DataFrame, indexed by critical point indexes
        fil_df: pd.DataFrame, indexed by filament indexes
        points_df: pd.DataFrame, indexed by point indexes
        specs: dict, with information such as image size, ndims...
        """
        self.critical_point = cp_df
        self.filament = fil_df
        self.point = point_df
        self.specs = specs

    def remove_lonely_cp(self):
        """ Remove critical point which are not conected to a filament.
        """
        self.critical_point['id'] = self.critical_point.index
        connected_cp = np.concatenate((self.filament.cp1.unique(),
                                       self.filament.cp2.unique()))

        unconnected_cp = np.setdiff1d(self.critical_point['id'], connected_cp)

        self.critical_point.drop(labels=unconnected_cp, axis=0, inplace=True)

        # update index in filament dataframe
        self.critical_point.reset_index(inplace=True)
        reset_cp1 = []
        reset_cp2 = []
        for i, (cp1, cp2) in self.filament[['cp1', 'cp2']].iterrows():
            reset_cp1.append(self.critical_point[
                             self.critical_point['id'] == cp1].index[0])
            reset_cp2.append(self.critical_point[
                             self.critical_point['id'] == cp2].index[0])
        self.filament['cp1'] = reset_cp1
        self.filament['cp2'] = reset_cp2

        self.critical_point.drop(labels='id', axis=1, inplace=True)

        warnings.warn(
            "specs dictionnary is not updated, ncrit value is not updated.")

    def remove_free_filament(self):
        """ Remove filament which have one free side.
        At the end there is a closed skeleton.
        """
        tip_cp = self.critical_point[self.critical_point['nfil'] == 1].index

        while len(tip_cp) > 0:

            # Find cp connected to 1 filament
            self.critical_point['id'] = self.critical_point.index
            self.filament['id'] = self.filament.index
            self.critical_point.drop(labels=tip_cp, axis=0, inplace=True)
            self.critical_point.reset_index(drop=True, inplace=True)
            remove_fil = []
            for i, (cp1, cp2) in self.filament[['cp1', 'cp2']].iterrows():
                if (not cp1 in self.critical_point['id'].to_numpy() or
                        not cp2 in self.critical_point['id'].to_numpy()):
                    remove_fil.append(i)
            self.filament.drop(labels=remove_fil, axis=0, inplace=True)
            self.filament.reset_index(drop=True, inplace=True)

            # update index in filament dataframe
            reset_cp1 = []
            reset_cp2 = []
            for i, (cp1, cp2) in self.filament[['cp1', 'cp2']].iterrows():
                reset_cp1.append(self.critical_point[
                                 self.critical_point['id'] == cp1].index[0])
                reset_cp2.append(self.critical_point[
                                 self.critical_point['id'] == cp2].index[0])
            self.filament['cp1'] = reset_cp1
            self.filament['cp2'] = reset_cp2

            # update nfil in self.critical_point
            i = self.filament.groupby(['cp1']).size().index
            val = self.filament.groupby(['cp1']).size().values
            self.critical_point.loc[i, 'nfil'] = val
            i = self.filament.groupby(['cp2']).size().index
            val = self.filament.groupby(['cp2']).size().values
            self.critical_point.loc[i, 'nfil'] = val
            tip_cp = self.critical_point[
                self.critical_point['nfil'] == 1].index

            self.point = self.point[self.point[
                'filament'].isin(self.filament['id'])]

            new_i = [self.filament[self.filament['id'] == old_i].index[0]
                     for old_i in self.point['filament']]
            self.point['filament'] = new_i

            self.critical_point.drop(labels='id', axis=1, inplace=True)
            self.filament.drop(labels='id', axis=1, inplace=True)
