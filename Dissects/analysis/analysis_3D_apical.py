import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from math import atan, atan2, acos
import scipy as sci
from skimage.draw import polygon

from ..utils.utils import pixel_to_um



def projected_points(face_df, edge_df, points_df, face, psi=0):
    points_df[['fz', 'fy', 'fx']] = face_df.loc[points_df.face.to_numpy()
                                                ][['fz', 'fy', 'fx']].to_numpy()
    points_df[['rz', 'ry', 'rx']] = points_df[['z', 'y', 'x']
                                              ].to_numpy() - points_df[['fz', 'fy', 'fx']].to_numpy()

    rel_pos = points_df.query(f"face == {face}")[["rx", "ry", "rz"]]
    rel_pos.index.name = "vert"
    _, _, rotation = np.linalg.svd(
        rel_pos.to_numpy().astype(float), full_matrices=False
    )
    if psi:
        rotation = np.dot(rotation_matrix(psi, [0, 0, 1]), rotation)
    rot_pos = pd.DataFrame(
        np.dot(rel_pos, rotation.T), index=rel_pos.index, columns=list('xyz')
    )
    return rot_pos


def morphology_analysis(segmentation,
                        area=True,
                        perimeter=True,
                        nb_neighbor=True,
                        aniso=True,
                        j_orientation=True,
                        angle_degree=False
                        ):
    
    segmentation.update_geom()
    segmentation.compute_normal()

    if area:
        # Approximate area
        segmentation.edge_df["sub_area"] = (
            np.linalg.norm(segmentation.edge_df[['nx', 'ny', 'nz']], axis=1) / 2
        )
        segmentation.face_df['area_approximate'] = _lvl_sum(
            segmentation.edge_df, segmentation.edge_df['sub_area'], 'face')

        # "real" area
        segmentation.face_df['area'] = 0
        for a in segmentation.face_df.index:
            if a != -1:
                points = projected_points(segmentation.face_df, segmentation.edge_df, segmentation.points_df, a)[
                    ['x', 'y']].to_numpy()
                hull = ConvexHull(points)
                segmentation.face_df.loc[a, 'area'] = hull.volume

    if perimeter:

        # Approximate perimeter
        segmentation.edge_df['length_approximate'] = np.sqrt(
            (segmentation.edge_df.sx-segmentation.edge_df.tx)**2+(segmentation.edge_df.sy-segmentation.edge_df.ty)**2+(segmentation.edge_df.sz-segmentation.edge_df.tz)**2).to_numpy()
        segmentation.face_df['perimeter_approximate'] = segmentation.edge_df.groupby('face')[
            'length_approximate'].sum()

        # Real perimeter
        segmentation.edge_df['length'] = 0
        for e in segmentation.edge_df.index:
            dist_ = 0
            for i in range(len(segmentation.points_df[segmentation.points_df.edge == e])-1):
                dist_ += np.sqrt((segmentation.points_df[segmentation.points_df.edge == e].iloc[i+1]['x'] - segmentation.points_df[segmentation.points_df.edge == e].iloc[i]['x'])**2 +
                                 (segmentation.points_df[segmentation.points_df.edge == e].iloc[i+1]['y'] - segmentation.points_df[segmentation.points_df.edge == e].iloc[i]['y'])**2 +
                                 (segmentation.points_df[segmentation.points_df.edge == e].iloc[i+1]['z'] - segmentation.points_df[segmentation.points_df.edge == e].iloc[i]['z'])**2)
            segmentation.edge_df.loc[e, 'length'] = dist_

        # face_df['perimeter'] = edge_df.groupby('face')[
        #     'length'].sum()
        # Perimeter with convexhull
        for a in segmentation.face_df.index:
            if a != -1:
                points = projected_points(segmentation.face_df, segmentation.edge_df, segmentation.points_df, a)[
                    ['x', 'y']].to_numpy()
                hull = ConvexHull(points)
                segmentation.face_df.loc[a, 'perimeter'] = hull.area

    if nb_neighbor:
        segmentation.face_df['nb_neighbor'] = segmentation.edge_df.groupby('face')[('srce')].count()

    if aniso:
        segmentation.points_df[['fz', 'fy', 'fx']] = segmentation.face_df.loc[segmentation.points_df.face.to_numpy()
                                                ][['fz', 'fy', 'fx']].to_numpy()
        for f in segmentation.face_df.index:
            if f != -1:
                points = segmentation.points_df[segmentation.points_df.face == f][['x', 'y', 'z']].to_numpy(
                                ).astype("float") - segmentation.points_df[segmentation.points_df.face == f][['fx', 'fy', 'fz']].to_numpy().astype("float")

                # Measure cell anisotropie
                u, s, vh = np.linalg.svd(points)

                svd =  np.concatenate((s, vh[0, :]))

                ocoords = ["orientation" + u for u in list('xyz')]
                segmentation.face_df.loc[f, ocoords] = svd[3:]

                s.sort()
                s = s[::-1]

                aniso = s[0] / s[1]

                segmentation.face_df.loc[f, 'major'] = s[0]
                segmentation.face_df.loc[f, 'minor'] = s[1]
                segmentation.face_df.loc[f, 'aniso'] = aniso

    if j_orientation:
        for e in segmentation.edge_df.index:
            points = segmentation.points_df[segmentation.points_df.edge == e][['x', 'y', 'z']].to_numpy(
                            ).astype("float") - segmentation.edge_df.loc[e][['sx','sy','sz']].to_numpy().astype("float")

            # Measure edge orientation according to xy plan
            segmentation.edge_df['orientation_xy'] = np.arctan2(segmentation.edge_df.ty-segmentation.edge_df.sy,
                                                                segmentation.edge_df.tx-segmentation.edge_df.sx)

            segmentation.edge_df['orientation_xz'] = np.arctan2(segmentation.edge_df.tz-segmentation.edge_df.sz,
                                                                segmentation.edge_df.tx-segmentation.edge_df.sx)

            segmentation.edge_df['orientation_yz'] = np.arctan2(segmentation.edge_df.tz-segmentation.edge_df.sz,
                                                                segmentation.edge_df.ty-segmentation.edge_df.sy)

            if angle_degree:
                segmentation.edge_df['orientation_xy'] = segmentation.edge_df['orientation_xy']*180/np.pi
                segmentation.edge_df['orientation_xz'] = segmentation.edge_df['orientation_xz']*180/np.pi
                segmentation.edge_df['orientation_yz'] = segmentation.edge_df['orientation_yz']*180/np.pi
    segmentation.face_df.drop(-1, axis=0, inplace=True)

def _lvl_sum(edge_df, df, lvl):
    df_ = df
    if isinstance(df, np.ndarray):
        df_ = pd.DataFrame(df, index=edge_df.index)
    elif isinstance(df, pd.Series):
        df_ = df.to_frame()
    elif lvl not in df.columns:
        df_ = df.copy()
    df_[lvl] = edge_df[lvl]
    return df_.groupby(lvl).sum()


def junction_intensity(image,
                       segmentation, 
                       dilation=3,
                       new_column='intensity'
                       ):
    """
    """

    segmentation.edge_df[new_column+'_mean'] = -1
    segmentation.edge_df[new_column+'_std'] = -1

    for e, val in segmentation.edge_df.iterrows():
        x_ = list(segmentation.points_df[segmentation.points_df.edge == e]['x_pix'].to_numpy().astype("int"))
        y_ = list(segmentation.points_df[segmentation.points_df.edge == e]['y_pix'].to_numpy().astype("int"))
        z_ = list(segmentation.points_df[segmentation.points_df.edge == e]['z_pix'].to_numpy().astype("int"))

        image_tmp = np.zeros(image.shape)

        image_tmp[z_, y_, x_] = 1

        s = sci.ndimage.generate_binary_structure(dilation, dilation)
        image_tmp = sci.ndimage.morphology.binary_dilation(
            image_tmp, s).astype(int)

        segmentation.edge_df.loc[e, new_column +
                    '_mean'] = np.mean(image[np.where(image_tmp == 1)])
        segmentation.edge_df.loc[e, new_column +
                    '_std'] = np.std(image[np.where(image_tmp == 1)])


def face_intensity(image,
                   segmentation,
                   thickness,
                   dilation,
                   new_column='intensity'):
    """

    Parameters
    ----------
    thickness : float, half thickness of the face in um

    Return
    ------
    all_enlarge_face_id: np.array, of all enlarge faces
    """
    image_no_junction = np.zeros(image.shape)
    image_no_junction[segmentation.points_df.z_pix.astype(int), 
                      segmentation.points_df.y_pix.astype(int), 
                      segmentation.points_df.x_pix.astype(int)] = 1
    s = sci.ndimage.generate_binary_structure(dilation, dilation)
    image_no_junction = ~sci.ndimage.morphology.binary_dilation(
                                     image_no_junction, s).astype(int)

    image_no_junction = image * image_no_junction
    
    
    all_enlarge_face_id = np.zeros(image.shape)
    segmentation.update_geom()
    segmentation.compute_normal()

    segmentation.face_df[new_column] = -1
    for f in segmentation.face_df.index:
        img_face = segmentation.enlarge_face_plane(face_id=f, thickness=thickness)
        
        z, y, x = np.where(img_face>0)
        all_enlarge_face_id[z,y,x] = f+1


        intensity_output = image_no_junction*img_face
        segmentation.face_df.loc[f, new_column] = np.mean(
            intensity_output[np.where(intensity_output > 0)])
    segmentation.face_df.drop(-1, axis=0, inplace=True)
    return all_enlarge_face_id





