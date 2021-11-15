import itertools
import logging

import numpy as np
import pandas as pd
import networkx as nx



from io import StringIO
from scipy import ndimage
from shapely.geometry import Polygon as shape_polygon
from skimage.draw import polygon, line_nd
from sklearn.neighbors import KDTree, BallTree

from ..utils.utils import pixel_to_um
from .segmentation import Segmentation


logger = logging.getLogger(name=__name__)
MAX_ITER = 10

default_image_specs = {
    "X_SIZE": 1,
    "Y_SIZE": 1,
    "Z_SIZE": 1
}


class Segmentation3D(Segmentation):

    def __init__(self):
        Segmentation.__init__(self)

    def __init__(self, skeleton):
        Segmentation.__init__(self, skeleton)

    def __init__(self, skeleton, specs):
        Segmentation.__init__(self, skeleton, specs)

    def find_vertex(self):
        """
        Extract vertex from DisperSE skeleton.
        As we use output of DisperSE for determined vertices. You need to process the breakdown
        part to have the correct placement of vertices.
        See http://www2.iap.fr/users/sousbie/web/html/index55a0.html?category/Quick-start for more information

        Parameters
        ----------
        skeleton : binary np.array; background=0, skeleton=1

        """

        save_column = list('xyz')[:self.skeleton.specs['ndims']]
        save_column.append('nfil')
        self.vert_df = self.skeleton.critical_point[self.skeleton.critical_point.nfil >= 3][save_column]

    def find_edge(self, half_edge=True):
        """
        Extract edges. Follow filaments from one vertex to another vertex, and define it as edge.

        Parameters
        ----------
        skeleton  : skel object
        vert_df   : DataFrame of vertices
        half_edge : boolean;

        """
        self.edge_df = pd.DataFrame(dtype='int')

        for i, val in self.vert_df.iterrows():
            start_cps = np.unique(self.skeleton.filament[(self.skeleton.filament.cp1 == i) | (
                self.skeleton.filament.cp2 == i)][['cp1', 'cp2']])
            for start in start_cps:
                sc = start

                filaments_id = []
                if sc != i:
                    # Get the first filament portion

                    filaments_id.append(self.skeleton.filament[((self.skeleton.filament.cp1 == i) & (self.skeleton.filament.cp2 == sc)) | (
                        (self.skeleton.filament.cp1 == sc) & (self.skeleton.filament.cp2 == i))].index[0])

                    previous_sc = i
                    previous_previous_sc = previous_sc
                    previous_sc = sc
                    while self.skeleton.critical_point.loc[sc]['nfil'] < 3:
                        tmp_sc = np.unique(self.skeleton.filament[(self.skeleton.filament.cp1 == previous_sc) | (
                            self.skeleton.filament.cp2 == previous_sc)][['cp1', 'cp2']])

                        for sc in tmp_sc:

                            if (sc != previous_previous_sc) and (
                                    sc != previous_sc):

                                filaments_id.append(self.skeleton.filament[((self.skeleton.filament.cp1 == previous_sc) & (self.skeleton.filament.cp2 == sc)) |
                                                                           ((self.skeleton.filament.cp1 == sc) & (self.skeleton.filament.cp2 == previous_sc))].index[0])

                                previous_previous_sc = previous_sc
                                previous_sc = sc
                                break

                    # Get coordinates from filament ids
                    pixel_x = self.skeleton.point[self.skeleton.point.filament.isin(
                        filaments_id)]['x'].to_numpy()
                    pixel_y = self.skeleton.point[self.skeleton.point.filament.isin(
                        filaments_id)]['y'].to_numpy()
                    pixel_z = self.skeleton.point[self.skeleton.point.filament.isin(
                        filaments_id)]['z'].to_numpy()
    #                 print(pixel_x)
                    edges = {'srce': i,
                             'trgt': sc,
                             'point_x': pixel_x,
                             'point_y': pixel_y,
                             'point_z': pixel_z,
                             'filaments': filaments_id}
                    self.edge_df = self.edge_df.append(
                        edges, ignore_index=True)

        self.edge_df.drop(self.edge_df[self.edge_df.srce == self.edge_df.trgt].index, inplace=True, )
        self.edge_df['min'] = np.min(self.edge_df[['srce', 'trgt']], axis=1)
        self.edge_df['max'] = np.max(self.edge_df[['srce', 'trgt']], axis=1)
        self.edge_df['srce'] = self.edge_df['min']
        self.edge_df['trgt'] = self.edge_df['max']
        self.edge_df.drop(['min', 'max'], axis=1, inplace=True)
        self.edge_df.drop_duplicates(inplace=True, subset=['srce', 'trgt'])
        self.edge_df.reset_index(drop=True, inplace=True)

        if half_edge:
            self.generate_half_edge()

    def generate_half_edge(self):
        """
        Generate half edge dataframe from edge and vert data frame

        """
        new_edge_df = pd.DataFrame(
            data=[np.zeros(len(self.edge_df.columns))], columns=self.edge_df.columns, dtype=object)

        for v0, data in self.vert_df.iterrows():
            va = [self.edge_df[(self.edge_df.srce == v0)]['trgt'].to_numpy()]
            va.append(
                self.edge_df[(self.edge_df.trgt == v0)]['srce'].to_numpy())
            va = [item for sublist in va for item in sublist]
            for v in va:
                dict_ = {'srce': v0,
                         'trgt': v,
                         }
                for c in self.edge_df.columns:
                    if (c != 'srce') & (c != 'trgt'):
                        dict_[c] = self.edge_df[
                            ((self.edge_df.srce == v0) & (self.edge_df.trgt == v))
                            |
                            (self.edge_df.trgt == v0) & (
                                self.edge_df.srce == v)
                        ][c].to_numpy()

                new_edge_df.loc[
                    np.max(new_edge_df.index) + 1
                ] = dict_

                dict_ = {'srce': v,
                         'trgt': v0,
                         }
                for c in self.edge_df.columns:
                    if (c != 'srce') & (c != 'trgt'):
                        dict_[c] = self.edge_df[
                            ((self.edge_df.srce == v0) & (self.edge_df.trgt == v))
                            |
                            (self.edge_df.trgt == v0) & (
                                self.edge_df.srce == v)
                        ][c].to_numpy()

                new_edge_df.loc[
                    np.max(new_edge_df.index) + 1
                ] = dict_

        new_edge_df.drop(index=0, axis=0, inplace=True)
        new_edge_df.drop_duplicates(inplace=True, subset=['srce', 'trgt'])
        new_edge_df.reset_index(drop=True, inplace=True)
        self.edge_df = new_edge_df.copy(deep=True)

    def find_cell(self):
        """
        Find face with graph theory

        Parameters
        ----------
        edge_df : DataFrame of edges

        Return
        ------
        face_df : DataFrame of faces
        edge_df : DataFrame of ordered edges
        """
        G = nx.from_pandas_edgelist(self.edge_df,
                                    source='srce',
                                    target='trgt',
                                    create_using=nx.Graph())
        all_faces = nx.minimum_cycle_basis(G)

        order_faces = [all_faces[0]]
        all_faces.remove(all_faces[0])

        find = True
        while (len(all_faces) > 0) or (find == False):
            find = False
            for i in range(len(all_faces)):
                ii = np.random.randint(0, len(all_faces))
                for j in range(len(order_faces)):
                    if len(set(order_faces[j]).intersection(all_faces[ii])) >= 2:
                        order_faces.append(all_faces[ii])
                        all_faces.remove(all_faces[ii])
                        find = True
                        break
                if find == True:
                    break

            if not find:
                break

        self.edge_df['face'] = -1

        cpt_face = 1

        for f in order_faces:

            edges = self.edge_df[(self.edge_df.srce.isin(f))
                                 & (self.edge_df.trgt.isin(f))]

            vert_order = [edges.iloc[0].srce]
            vert_order.append(edges.iloc[0].trgt)
            for i in range(len(edges)):
                vert_order.append(edges[(edges.srce == vert_order[-1]) &
                                        (edges.trgt != vert_order[-2])]['trgt'].to_numpy()[0])
                if vert_order[0] == vert_order[-1]:
                    break
            if self.check_poly_is_valid(vert_order):
                edge = []
                for i in range(len(vert_order) - 1):
                    edge.append(self.edge_df[(self.edge_df.srce == vert_order[i]) & (
                        self.edge_df.trgt == vert_order[i + 1])].index.to_numpy()[0])

                if len(np.unique(self.edge_df.loc[edge]['face'].to_numpy())) == 1:
                    self.edge_df.loc[edge, 'face'] = cpt_face

                else:
                    vert_order = np.flip(vert_order)
                    edge = []
                    for i in range(len(vert_order) - 1):
                        edge.append(self.edge_df[(self.edge_df.srce == vert_order[i]) & (
                            self.edge_df.trgt == vert_order[i + 1])].index.to_numpy()[0])
                    if len(np.unique(self.edge_df.loc[edge]['face'].to_numpy())) == 1:
                        self.edge_df.loc[edge, 'face'] = cpt_face
                    else:
                        logger.warning("there is a problem")
                        # print(f)
                        # print(tmp)
                        # print(tmp_e)
                        # print((edge_df.loc[edge]['face'].to_numpy()))
                        # print(edge)

                cpt_face += 1
            else:
                logger.warning("Self intersect polygon")

        self.edge_df.drop(self.edge_df[self.edge_df['face'] == -1].index, inplace=True)
        self.edge_df.reset_index(drop=True, inplace=True)
        self.face_df = pd.DataFrame(index=np.sort(self.edge_df.face.unique()))

    def find_points(self):
        self.points_df = pd.DataFrame(
            columns=['x_pix', 'y_pix', 'z_pix', 'edge', 'face'])

        for e, val in self.edge_df.iterrows():
            for i in range(len(val['point_x'][0])):
                dict_ = {'x_pix': val['point_x'][0][i],
                         'y_pix': val['point_y'][0][i],
                         'z_pix': val['point_z'][0][i],
                         'edge': e,
                         'face': val.face}
                self.points_df = self.points_df.append(
                    dict_, ignore_index=True)

    def generate_segmentation(self,
                              clean=True,
                              **kwargs
                              ):
        """
        Generate three dataframe from binary skeleton image which contains informations
        about faces, edges and vertex
        Put coordinates of vertex into um. Keep pixel position in "xyz_pix" columns.

        Parameters
        ----------
        skeleton : binary np.array; background=0, skeleton=1

        Return
        ------
        face_df : DataFrame
        edge_df : DataFrame
        vert_df : DataFrame
        """

        self.find_vertex()
        self.find_edge(half_edge=True)
        self.find_cell()
        self.find_points()

        self.vert_df['x_pix'] = self.vert_df['x']
        self.vert_df['y_pix'] = self.vert_df['y']
        self.vert_df['z_pix'] = self.vert_df['z']

        pixel_to_um(self.vert_df, self.specs, [
                    'x_pix', 'y_pix', 'z_pix'], list('xyz'))
        pixel_to_um(self.points_df, self.specs, [
                    'x_pix', 'y_pix', 'z_pix'], list('xyz'))

        # Mark face in the border
        # It remove more cell than expected...
        self.edge_df['opposite'] = -1
        for e, val in self.edge_df.iterrows():
            tmp = self.edge_df[(self.edge_df.srce == val.trgt) & (
                self.edge_df.trgt == val.srce)].index.to_numpy()
            if len(tmp) > 0:
                self.edge_df.loc[e, 'opposite'] = tmp

        self.face_df["border"] = 0
        self.face_df.loc[self.edge_df[self.edge_df.opposite == -1]
                    ['face'].to_numpy(), 'border'] = 1
        
        self.update_geom()
        self.compute_normal()

        self.face_df.drop(-1, axis=0, inplace=True)


    def check_poly_is_valid(self, vert_order):
        """
        Verifie si le polygon s'auto intersecte ou pas.
        """
        

        center = self.vert_df.loc[vert_order][list('xyz')].mean()
        rel_srce_pos = np.array([(self.vert_df.loc[vert_order, 'x'] - center.x).to_numpy(),
                                 (self.vert_df.loc[vert_order, 'y'] -
                                  center.y).to_numpy(),
                                 (self.vert_df.loc[vert_order, 'z'] - center.z).to_numpy()])

        rel_srce_pos = pd.DataFrame(rel_srce_pos.T, columns=list("xyz"))

        _, _, rotation = np.linalg.svd(
            rel_srce_pos.astype(float), full_matrices=False
        )

        rot_pos = np.dot(rel_srce_pos, rotation.T)
        np.arctan2(rot_pos[:, 1], rot_pos[:, 0])

        coords = []
        for i in range(len(rot_pos)):
            coords.append((rot_pos[i][0], rot_pos[i][1]))
        poly = shape_polygon(coords)
        return poly.is_valid



    def image_identity_vertex(self, binary=True, dilation_width=3):
        tiff_vertex = np.zeros([self.specs['z_shape'], 
                                self.specs['y_shape'],
                                self.specs['x_shape']])

        if binary:
            tiff_vertex[self.vert_df.z_pix.to_numpy().astype(int),
                        self.vert_df.y_pix.to_numpy().astype(int),
                        self.vert_df.x_pix.to_numpy().astype(int)] = 1
            if dilation_width!=0:
                s = ndimage.generate_binary_structure(dilation_width, dilation_width)
                tiff_vertex = ndimage.morphology.binary_dilation(tiff_vertex, structure=s)
        else : 
            if dilation_width !=0:
                s = ndimage.generate_binary_structure(dilation_width, dilation_width)
                for v in self.vert_df.index:
                    tmp = np.zeros([self.specs['z_shape'], 
                                    self.specs['y_shape'],
                                    self.specs['x_shape']])
                    tmp[self.vert_df.loc[v, 'z_pix'].astype(int), 
                        self.vert_df.loc[v, 'y_pix'].astype(int), 
                        self.vert_df.loc[v, 'x_pix'].astype(int)] = 1

                    tmp = ndimage.morphology.binary_dilation(tmp, structure=s).astype(int)
                    pos = np.where(tmp==1)
                    tiff_vertex[pos] = v+1

            else : 
                for v in self.vert_df.index:
                    tiff_vertex[self.vert_df.loc[v, 'z_pix'].astype(int), 
                                self.vert_df.loc[v, 'y_pix'].astype(int), 
                                self.vert_df.loc[v, 'x_pix'].astype(int)] = v+1


        return tiff_vertex

    def image_identity_junction(self, dilation_width=3, aleatory=False):
        tiff_junction = np.zeros([self.specs['z_shape'], 
                                  self.specs['y_shape'],
                                  self.specs['x_shape']])
        unique_value_edge = np.unique(self.edge_df.index)
        replace_value_edge = unique_value_edge.copy()
        if aleatory:
            replace_value_edge = np.random.randint(1, 2**16, len(unique_value_edge))

        if dilation_width!=0:
            s = ndimage.generate_binary_structure(dilation_width, dilation_width)
            for e in unique_value_edge:
                tmp = np.zeros([self.specs['z_shape'], 
                                self.specs['y_shape'],
                                self.specs['x_shape']])
                tmp[list(self.edge_df.loc[unique_value_edge[e], 'point_z'])[0].astype(int),
                    list(self.edge_df.loc[unique_value_edge[e], 'point_y'])[0].astype(int),
                    list(self.edge_df.loc[unique_value_edge[e], 'point_x'])[0].astype(int)] = 1
                
                tmp = ndimage.morphology.binary_dilation(tmp, structure=s).astype(int)
                pos = np.where(tmp==1)
                tiff_junction[pos] = replace_value_edge[e]

        else : 
            for e in unique_value_edge:
                tiff_junction[list(self.edge_df.loc[unique_value_edge[e], 'point_z'])[0].astype(int),
                              list(self.edge_df.loc[unique_value_edge[e], 'point_y'])[0].astype(int),
                              list(self.edge_df.loc[unique_value_edge[e], 'point_x'])[0].astype(int)] = replace_value_edge[e]
                
        return tiff_junction

    def image_analyse_junction(self, column, normalize=False, normalize_max=None, dilation_width=3, border=True):
        tiff_junction = np.zeros([self.specs['z_shape'], 
                                  self.specs['y_shape'],
                                  self.specs['x_shape']])
        
        if normalize:
            if normalize_max is None:
                self.edge_df[column+'norm'] = self.edge_df[column]/np.max(self.edge_df[column])
            else:
                self.edge_df[column+'norm'] = self.edge_df[column]/normalize_max
            column = column+'norm'

        if border: 
            edge_id = self.edge_df.index.to_numpy()
        else:
            edge_id = self.edge_df[self.edge_df.opposite!=-1].index.to_numpy()

        for e in edge_id:
            tmp = np.zeros([self.specs['z_shape'], 
                              self.specs['y_shape'],
                              self.specs['x_shape']])
            x_ = list(self.points_df[self.points_df.edge==e]['x_pix'].to_numpy().astype(int))
            y_ = list(self.points_df[self.points_df.edge==e]['y_pix'].to_numpy().astype(int))
            z_ = list(self.points_df[self.points_df.edge==e]['z_pix'].to_numpy().astype(int))
            #(val.myosin_intensity_mean/np.max(edge_df.myosin_intensity_mean)*255).astype('int')
            
            if dilation_width!=0:
                s = ndimage.generate_binary_structure(dilation_width, dilation_width)
                tmp[z_, y_, x_] = 1
                tmp = ndimage.morphology.binary_dilation(tmp, structure=s).astype(int)
                pos = np.where(tmp==1)
                tiff_junction[pos] = self.edge_df.loc[e, column]
            else:
                tiff_junction[z_, y_, x_] = self.edge_df.loc[e, column]
    
        return tiff_junction


    def image_identity_face(self, aleatory=False, thickness=0.5):
        tiff_face = np.zeros([self.specs['z_shape'], 
                              self.specs['y_shape'],
                              self.specs['x_shape']])
        unique_value_face = np.unique(self.face_df.index)
        replace_value_face = unique_value_face.copy()
        if aleatory:
            replace_value_face = np.random.randint(1, 2**16, len(unique_value_face))

        for f in range(len(unique_value_face)):
            tmp = self.enlarge_face_plane(unique_value_face[f], thickness)
            pos = np.where(tmp==1)
            tiff_face[pos] = replace_value_face[f]

        return tiff_face


    def image_analyse_face(self, column, normalize=False, normalize_max=None, thickness=0.5, border=True):
        tiff_face = np.zeros([self.specs['z_shape'], 
                              self.specs['y_shape'],
                              self.specs['x_shape']])

        if normalize:
            if normalize_max is None:
                self.face_df[column+'norm'] = self.face_df[column]/np.max(self.face_df[column])
            else:
                self.face_df[column+'norm'] = self.face_df[column]/normalize_max
            column = column+'norm'


        if border:
            face_id = self.face_df.index.to_numpy()
        else:
            face_id = self.face_df[self.face_df.border==0].index.to_numpy()

        for f in face_id:
            tmp = self.enlarge_face_plane(f, thickness=thickness)
            face_position = np.where(tmp==1)
            
            tiff_face[face_position] = self.face_df.loc[f, column]

        return tiff_face


    def image_aniso(self, normalize=True, normalize_max=None, factor=2, dilation_width=3):

        tiff_aniso = np.zeros([self.specs['z_shape'], 
                              self.specs['y_shape'],
                              self.specs['x_shape']])
        if normalize:
            if normalize_max is None:
                self.face_df['aniso_norm'] = self.face_df['aniso']/np.max(self.face_df['aniso'])
            else:
                self.face_df['aniso_norm'] = self.face_df['aniso']/normalize_max

        startx = ((self.face_df['fx'] - factor*self.face_df['aniso_norm']*self.face_df['orientationx'])/self.specs['x_size']).to_numpy().astype(int)
        starty = ((self.face_df['fy'] - factor*self.face_df['aniso_norm']*self.face_df['orientationy'])/self.specs['y_size']).to_numpy().astype(int)
        startz = ((self.face_df['fz'] - factor*self.face_df['aniso_norm']*self.face_df['orientationz'])/self.specs['z_size']).to_numpy().astype(int)
        endx = ((self.face_df['fx'] + factor*self.face_df['aniso_norm']*self.face_df['orientationx'])/self.specs['x_size']).to_numpy().astype(int)
        endy = ((self.face_df['fy'] + factor*self.face_df['aniso_norm']*self.face_df['orientationy'])/self.specs['y_size']).to_numpy().astype(int)
        endz = ((self.face_df['fz'] + factor*self.face_df['aniso_norm']*self.face_df['orientationz'])/self.specs['z_size']).to_numpy().astype(int)
        c = (self.face_df['aniso_norm']*255/np.max(self.face_df['aniso_norm'])).to_numpy()
        
        for i in range(len(startx)):

            tmp_image = np.zeros([self.specs['z_shape'], 
                                  self.specs['y_shape'],
                                  self.specs['x_shape']])

            coords = line_nd((startx[i], starty[i], startz[i]),
                             (endx[i], endy[i], endz[i]), 
                             endpoint=True,
                             integer=True)

            tmp_image[coords[2],
                       coords[1], 
                       coords[0]] = 1
            #enlarge
            if dilation_width!=0:
                s = ndimage.generate_binary_structure(dilation_width, dilation_width)
                tmp_image = ndimage.morphology.binary_dilation(tmp_image, structure=s)

            pos = np.where(tmp_image==1)
            tiff_aniso[pos] = c[i]

        return tiff_aniso


    def update_geom(self):
        self.edge_df[['sx', 'sy', 'sz']] = self.vert_df.loc[self.edge_df.srce,
                                                  list('xyz')].to_numpy()
        self.edge_df[['tx', 'ty', 'tz']] = self.vert_df.loc[self.edge_df.trgt,
                                                  list('xyz')].to_numpy()

        self.face_df[['fx', 'fy', 'fz']] = 0
        self.face_df['fx'] = self.edge_df.groupby('face').mean()['sx']
        self.face_df['fy'] = self.edge_df.groupby('face').mean()['sy']
        self.face_df['fz'] = self.edge_df.groupby('face').mean()['sz']



    def compute_normal(self):
        self.face_df.loc[-1] = np.zeros((self.face_df.shape[1]))

        # update centroid face
        #face_df[list('xyz')] = edge_df.groupby("face")[['sx','sy','sz']].mean()
        self.edge_df[['fz', 'fy', 'fx']] = self.face_df.loc[self.edge_df.face.to_numpy()
                                                  ][['fz', 'fy', 'fx']].to_numpy()
        self.edge_df[['dz', 'dy', 'dx']] = self.edge_df[['tz', 'ty', 'tx']
                                              ].to_numpy() - self.edge_df[['sz', 'sy', 'sx']].to_numpy()
        self.edge_df[['rz', 'ry', 'rx']] = self.edge_df[['sz', 'sy', 'sx']
                                              ].to_numpy() - self.edge_df[['fz', 'fy', 'fx']].to_numpy()
        r_ij = self.edge_df[['dz', 'dy', 'dx']].to_numpy()
        r_ai = self.edge_df[['rz', 'ry', 'rx']].to_numpy()
        normals = np.cross(r_ai, r_ij)
        self.edge_df[['nz', 'ny', 'nx']] = normals


    def enlarge_face_plane(self, 
                           face_id,
                           thickness=0.5,
                            ):

        # normal normalisé ? 
        n = (np.mean(self.edge_df[self.edge_df['face'] == face_id][['nx', 'ny', 'nz']])/np.linalg.norm(
             np.mean(self.edge_df[self.edge_df['face'] == face_id][['nx', 'ny', 'nz']]))).to_numpy()
        
        # list points in um   
        xx = self.points_df[self.points_df.face==face_id]['x']
        yy = self.points_df[self.points_df.face==face_id]['y']
        zz = self.points_df[self.points_df.face==face_id]['z']
        
        # Find the top and bottom position according to face plane in um
        top = np.array((xx, yy, zz)).flatten(order='F').reshape((len(xx), 3)) + thickness*n
        bottom = np.array((xx, yy, zz)).flatten(order='F').reshape((len(xx), 3)) - thickness*n
        top = pd.DataFrame(top, columns=[list('xyz')])
        bottom = pd.DataFrame(bottom, columns=[list('xyz')])

        # Convert um position in pixel position
        top['x_pix'] = (top['x']/self.specs['x_size']).astype('int')
        top['y_pix'] = (top['y']/self.specs['y_size']).astype('int')
        top['z_pix'] = (top['z']/self.specs['z_size']).astype('int')
        bottom['x_pix'] = (bottom['x']/self.specs['x_size']).astype('int')
        bottom['y_pix'] = (bottom['y']/self.specs['y_size']).astype('int')
        bottom['z_pix'] = (bottom['z']/self.specs['z_size']).astype('int')

        # Replace value which exceed boundary to image border value
        top['x_pix'] = np.where((top['x_pix']<0), 0, top['x_pix'])
        top['y_pix'] = np.where((top['y_pix']<0), 0, top['y_pix'])
        top['z_pix'] = np.where((top['z_pix']<0), 0, top['z_pix'])
        bottom['x_pix'] = np.where((bottom['x_pix']<0), 0, bottom['x_pix'])
        bottom['y_pix'] = np.where((bottom['y_pix']<0), 0, bottom['y_pix'])
        bottom['z_pix'] = np.where((bottom['z_pix']<0), 0, bottom['z_pix'])
        
        top['x_pix'] = np.where((top['x_pix']>=self.specs['x_shape']), self.specs['x_shape']-1, top['x_pix'])
        top['y_pix'] = np.where((top['y_pix']>=self.specs['y_shape']), self.specs['y_shape']-1, top['y_pix'])
        top['z_pix'] = np.where((top['z_pix']>=self.specs['z_shape']), self.specs['z_shape']-1, top['z_pix'])
        bottom['x_pix'] = np.where((bottom['x_pix']>=self.specs['x_shape']), self.specs['x_shape']-1, bottom['x_pix'])
        bottom['y_pix'] = np.where((bottom['y_pix']>=self.specs['y_shape']), self.specs['y_shape']-1, bottom['y_pix'])
        bottom['z_pix'] = np.where((bottom['z_pix']>=self.specs['z_shape']), self.specs['z_shape']-1, bottom['z_pix'])
        
        
        img_plane = np.zeros([self.specs['z_shape'], self.specs['y_shape'],self.specs['x_shape']])
        try:
            # top plane
            for i, data in top.iterrows():
                img_plane[int(data.z_pix), int(data.y_pix), int(data.x_pix)] = 1
            # bottom plane
            for i, data in bottom.iterrows():
                img_plane[int(data.z_pix), int(data.y_pix), int(data.x_pix)] = 1
            # middle plane
            for i, data in self.vert_df.loc[self.edge_df[self.edge_df.face == face_id].srce.to_numpy()].iterrows():
                img_plane[int(data.z_pix), int(data.y_pix), int(data.x_pix)] = 1
       
        except Exception as ex:
                print(ex)
                pass


        pts = np.concatenate((np.where(img_plane == 1)[0], np.where(img_plane == 1)[
                             1], np.where(img_plane == 1)[2])).reshape(3, len(np.where(img_plane == 1)[0]))
        pts = pts.flatten(order='F').reshape(len(np.where(img_plane == 1)[0]), 3)
        
        test = pd.DataFrame(pts, columns=list('zyx'))

        for z_ in np.unique(test.z):
            poly_points = test[test.z == z_][list('xy')]
            if poly_points.shape[0] > 1:
                mask = np.zeros(
                    (img_plane.shape[2], img_plane.shape[1]), dtype=np.uint8)
                r = poly_points.to_numpy().flatten()[0::2]
                c = poly_points.to_numpy().flatten()[1::2]
                rr, cc = polygon(r, c)
                mask[rr, cc] = 1
                xx, yy = np.where(mask != 0)
                for i in range(len(xx)):
                    img_plane[int(z_), int(yy[i]), int(xx[i])] = 1
            else:
                img_plane[int(z_), int(poly_points.y), int(poly_points.x)] = 1

        for y_ in np.unique(test.y):
            poly_points = test[test.y == y_][list('xz')]
            if poly_points.shape[0] > 1:
                mask = np.zeros(
                    (img_plane.shape[2], img_plane.shape[0]), dtype=np.uint8)
                r = poly_points.to_numpy().flatten()[0::2]
                c = poly_points.to_numpy().flatten()[1::2]
                rr, cc = polygon(r, c)
                mask[rr, cc] = 1
                xx, zz = np.where(mask != 0)
                for i in range(len(xx)):
                    img_plane[int(zz[i]), int(y_), int(xx[i])] = 1
            else:
                img_plane[int(poly_points.z), int(y_), int(poly_points.x)] = 1

        for x_ in np.unique(test.x):
            poly_points = test[test.x == x_][list('yz')]
            if poly_points.shape[0] > 1:
                mask = np.zeros(
                    (img_plane.shape[1], img_plane.shape[0]), dtype=np.uint8)
                r = poly_points.to_numpy().flatten()[0::2]
                c = poly_points.to_numpy().flatten()[1::2]
                rr, cc = polygon(r, c)
                mask[rr, cc] = 1
                yy, zz = np.where(mask != 0)
                for i in range(len(yy)):
                    img_plane[int(zz[i]), int(yy[i]), int(x_)] = 1
            else:
                img_plane[int(poly_points.z), int(poly_points.y), int(x_)] = 1

        return img_plane


def junctions_length(skeleton, pixel_size, clean=True):
    """
    Measure the length of the junctions.
    Follow each junction from one vertex to the next one, defines it as an edge and calcul its length.

    Parameters
    ----------
    skel : skeleton object, breakdowned, smoothed and cleaned

    Return
    ------
    vert_df : DataFrame
    df_junc : Dataframe
    """

    skeleton.critical_point['z'] = skeleton.critical_point['z'] * \
        pixel_size['Z_SIZE'] / pixel_size['X_SIZE']
    skeleton.point['z'] = skeleton.point['z'] * \
        pixel_size['Z_SIZE'] / pixel_size['X_SIZE']

    if not clean:
        skeleton.critical_point['id_original'] = skeleton.critical_point.index
        skeleton.filament['id_original'] = skeleton.filament.index

    # now try and do it for each node
    idx_nflsup3 = np.where(skeleton.critical_point.nfil >= 3)[0]

    lf = []
    end_fil = []
    lcp = []
    il = 0
    for idx_depart in idx_nflsup3:

        i_original_depart = skeleton.critical_point.iloc[idx_depart]['id_original']

        mask_clean = np.isin(skeleton.cp_fil_info.iloc[i_original_depart]['destcritid'],
                             skeleton.critical_point.id_original)
        cp_filId = list(
            np.array(skeleton.cp_fil_info.iloc[i_original_depart]['fillId'])[mask_clean])
        # filament indices
        idx_fils = [np.where(skeleton.filament['id_original'] ==
                             cp_filId[i])[0][0]
                    for i in range(len(cp_filId))]

        # destination cp indices
        cp_destId = list(
            np.array(skeleton.cp_fil_info.iloc[i_original_depart]['destcritid'])[mask_clean])

        # destination cp indices
        idx_destcrits = [np.where(skeleton.critical_point['id_original'] ==
                                  cp_destId[i])[0][0]
                         for i in range(len(cp_destId))]

        for inum, fi, ci in zip(range(len(idx_fils)), idx_fils, idx_destcrits):
            dead_end = False
            if not np.isin(fi, end_fil):
                list_fils = [fi]
                list_cp = [idx_depart, ci]
                idx_thiscp = ci
                ior_previous = i_original_depart
                dest_cps = skeleton.cp_fil_info.iloc[skeleton.critical_point.iloc[idx_thiscp]['id_original']]
                mask_clean = np.isin(
                    dest_cps['destcritid'], skeleton.critical_point.id_original)
                dest_cps['destcritid'] = list(
                    np.array(dest_cps['destcritid'])[mask_clean])
                dest_cps['fillId'] = list(
                    np.array(dest_cps['fillId'])[mask_clean])
                if np.sum(
                        np.isin(np.array(dest_cps['destcritid']), ior_previous)) > 1:
                    print('PROBLEM')
                    break
                if (len(np.isin(np.array(dest_cps['destcritid']), ior_previous)) == 1 and
                        np.isin(np.array(dest_cps['destcritid']), ior_previous) == [True]):
                    #print('dead end outside while')
                    lf.append(list_fils)
                    end_fil.append(list_fils[-1])
                    lcp.append(list_cp)
                    #dead_end = True
                    continue
                else:
                    next_cp = np.array(dest_cps['destcritid'])[~np.isin(
                        np.array(dest_cps['destcritid']), ior_previous)][0]
                    idx_nextcp = np.where(np.isin(skeleton.critical_point['id_original'],
                                                  next_cp))[0][0]

                iwhile = 0
                while skeleton.critical_point.iloc[idx_thiscp]['nfil'] < 3:
                    ior_previous = skeleton.critical_point.iloc[idx_thiscp]['id_original']
                    idx_thiscp = idx_nextcp
                    ior_current = skeleton.critical_point.iloc[idx_nextcp]['id_original']

                    mask_clean = np.isin(skeleton.cp_fil_info.iloc[ior_previous]['destcritid'],
                                         skeleton.critical_point.id_original)
                    previous_cp_filId = list(
                        np.array(skeleton.cp_fil_info.iloc[ior_previous]['fillId'])[mask_clean])

                    idx_fili = np.array([np.where(skeleton.filament['id_original'] ==
                                                  skeleton.cp_fil_info.iloc[ior_previous]['fillId'][i])[0][0]
                                         for i in range(len(skeleton.cp_fil_info.iloc[ior_previous]['fillId']))])
                    next_fil = idx_fili[~np.isin(
                        idx_fili, np.array(list_fils))][0]
                    list_fils.append(next_fil)
                    list_cp.append(idx_nextcp)

                    dest_cps = skeleton.cp_fil_info.iloc[skeleton.critical_point.iloc[idx_nextcp]['id_original']]
                    mask_clean = np.isin(
                        dest_cps['destcritid'], skeleton.critical_point.id_original)
                    dest_cps['destcritid'] = list(
                        np.array(dest_cps['destcritid'])[mask_clean])
                    dest_cps['fillId'] = list(
                        np.array(dest_cps['fillId'])[mask_clean])
                    if np.sum(
                            np.isin(np.array(dest_cps['destcritid']), ior_previous)) > 1:
                        print('PROBLEM')
                        break
                    if (len(np.isin(np.array(dest_cps['destcritid']), ior_previous)) == 1 and
                            np.isin(np.array(dest_cps['destcritid']), ior_previous) == [True]):
                        lf.append(list_fils)
                        end_fil.append(list_fils[-1])
                        lcp.append(list_cp)
                        #print('dead end inside while')
                        dead_end = True
                        break
                    else:
                        next_cp = np.array(dest_cps['destcritid'])[
                            ~np.isin(np.array(dest_cps['destcritid']), ior_previous)][0]
                        idx_nextcp = np.where(np.isin(skeleton.critical_point['id_original'],
                                                      next_cp))[0][0]

                    iwhile += 1
                if not dead_end:
                    lf.append(list_fils)
                    end_fil.append(list_fils[-1])
                    lcp.append(list_cp)

    Junctions = np.empty(len(lf), dtype='object')
    junc_points = np.empty(len(lf), dtype='object')
    junc_cp_ends = np.empty(len(lf), dtype='object')
    junc_cp_ends_srce = np.empty(len(lf), dtype='object')
    junc_cp_ends_trgt = np.empty(len(lf), dtype='object')
    length = np.zeros(len(lf))

    for ijunc in range(len(lf)):
        Junctions[ijunc] = []
        ifil = 0
        for fil in lf[ijunc]:
            if ifil == 0:
                ppoint = np.array(
                    skeleton.point[skeleton.point['filament'] == fil][['x', 'y', 'z']])
                Junctions[ijunc].append(ppoint)
            if ifil == 1:
                ppoint_before = ppoint
                ppoint = np.array(
                    skeleton.point[skeleton.point['filament'] == fil][['x', 'y', 'z']])
                ppoint1 = ppoint
                if np.all(ppoint[-1] == ppoint_before[0]):
                    # flip 0+1
                    Junctions[ijunc][0] = np.flip(Junctions[ijunc][0], axis=0)
                    ppoint1 = np.flip(ppoint, axis=0)
                    Junctions[ijunc].append(ppoint1)
                elif np.all(ppoint[0] == ppoint_before[0]):
                    # flip 0
                    Junctions[ijunc][0] = np.flip(Junctions[ijunc][0], axis=0)
                    Junctions[ijunc].append(ppoint)
                elif np.all(ppoint[-1] == ppoint_before[-1]):
                    # flip 1
                    ppoint1 = np.flip(ppoint, axis=0)
                    Junctions[ijunc].append(ppoint1)
                else:
                    Junctions[ijunc].append(ppoint)
                ppoint = ppoint1
            if ifil > 1:
                ppoint_before = ppoint
                ppoint = np.array(
                    skeleton.point[skeleton.point['filament'] == fil][['x', 'y', 'z']])
                ppoint1 = ppoint
                if np.all(ppoint[-1] == ppoint_before[-1]):
                    # flip 1
                    ppoint1 = np.flip(ppoint, axis=0)
                    Junctions[ijunc].append(ppoint1)
                else:
                    Junctions[ijunc].append(ppoint)
                ppoint = ppoint1
            ifil += 1
            junc_points[ijunc] = np.concatenate(Junctions[ijunc])

            junc_cp_ends[ijunc] = [lcp[ijunc][0], lcp[ijunc][-1]]
            junc_cp_ends_srce[ijunc] = junc_cp_ends[ijunc][0]
            junc_cp_ends_trgt[ijunc] = junc_cp_ends[ijunc][1]
            # junc_cp_ends[ijunc] = [float_vert_df.index[np.where(idx_nflsup3 == lcp[ijunc][0])[0][0]],
            # float_vert_df.index[np.where(idx_nflsup3 ==
            # lcp[ijunc][-1])[0][0]]]
            length[ijunc] = np.sum(np.sqrt(np.sum(
                ((np.roll(junc_points[ijunc], 1, axis=0) -
                  junc_points[ijunc])[1:])**2, axis=1)))

    df_junc = pd.DataFrame(data={'srce': junc_cp_ends_srce,
                                 'trgt': junc_cp_ends_trgt,
                                 'points_coords': junc_points,
                                 'points_coords_binaire': [junc_points[ijunc].astype(int)
                                                           for ijunc in range(len(junc_points))],
                                 'length_AU': length,
                                 'length_um': length * pixel_size['X_SIZE']

                                 })

    skeleton.critical_point['z'] = skeleton.critical_point['z'] * \
        pixel_size['X_SIZE'] / pixel_size['Z_SIZE']
    skeleton.point['z'] = skeleton.point['z'] * \
        pixel_size['X_SIZE'] / pixel_size['Z_SIZE']

    return df_junc


def assign_length(df_junc, edge_df, vert_df):
    edge_df["length"] = np.nan

    for i in range(len(edge_df)):
        srce_xyz = (np.around(vert_df.loc[edge_df.loc[i].srce].x_pix, 1),
                    np.around(vert_df.loc[edge_df.loc[i].srce].y_pix, 1),
                    np.around(vert_df.loc[edge_df.loc[i].srce].z_pix, 1))
    
        trgt_xyz = (np.around(vert_df.loc[edge_df.loc[i].trgt].x_pix, 1),
                    np.around(vert_df.loc[edge_df.loc[i].trgt].y_pix, 1),
                    np.around(vert_df.loc[edge_df.loc[i].trgt].z_pix, 1))
        junc_i1 = np.array([srce_xyz, trgt_xyz])
        junc_i2 = np.array([trgt_xyz, srce_xyz])

        for ind in range(len(df_junc)):
            junc_ind1 = np.array([df_junc.s_xyz[ind], df_junc.t_xyz[ind]])
            junc_ind2 = np.array([df_junc.t_xyz[ind], df_junc.s_xyz[ind]])
            if np.all(junc_i1 == junc_ind1) or np.all(junc_i1 == junc_ind2) or np.all(
                    junc_i2 == junc_ind2) or np.all(junc_i2 == junc_ind1):
                edge_df["length"][i] = df_junc['length_um'][ind]
