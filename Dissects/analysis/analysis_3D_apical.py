import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from math import atan, atan2, acos
import scipy as sci
from skimage.draw import polygon

from ..utils.utils import pixel_to_um


def update_geom(face_df, edge_df, vert_df):
    edge_df[['sx', 'sy', 'sz']] = vert_df.loc[edge_df.srce,
                                              list('xyz')].to_numpy()
    edge_df[['tx', 'ty', 'tz']] = vert_df.loc[edge_df.trgt,
                                              list('xyz')].to_numpy()

    face_df[['fx', 'fy', 'fz']] = 0
    for i in face_df.index:
        face_df.loc[i, 'fx'] = edge_df[edge_df.face == i].mean()['sx']
        face_df.loc[i, 'fy'] = edge_df[edge_df.face == i].mean()['sy']
        face_df.loc[i, 'fz'] = edge_df[edge_df.face == i].mean()['sz']


def compute_normal(face_df, edge_df, vert_df):
    face_df.loc[-1] = np.zeros((face_df.shape[1]))

#     # update centroid face
#     face_df[list('xyz')] = edge_df.groupby("face")[['sx','sy','sz']].mean()
    edge_df[['fz', 'fy', 'fx']] = face_df.loc[edge_df.face.to_numpy()
                                              ][['fz', 'fy', 'fx']].to_numpy()
    edge_df[['dz', 'dy', 'dx']] = edge_df[['tz', 'ty', 'tx']
                                          ].to_numpy() - edge_df[['sz', 'sy', 'sx']].to_numpy()
    edge_df[['rz', 'ry', 'rx']] = edge_df[['sz', 'sy', 'sx']
                                          ].to_numpy() - edge_df[['fz', 'fy', 'fx']].to_numpy()
    r_ij = edge_df[['dz', 'dy', 'dx']].to_numpy()
    r_ai = edge_df[['rz', 'ry', 'rx']].to_numpy()
    normals = np.cross(r_ai, r_ij)
    edge_df[['nz', 'ny', 'nx']] = normals


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


def morphology_analysis(face_df,
                        edge_df,
                        vert_df,
                        points_df,
                        area=True,
                        perimeter=True,
                        nb_neighbor=True,
                        aniso=True,
                        j_orientation=True,
                        angle_degree=False
                        ):

    update_geom(face_df, edge_df, vert_df)
    compute_normal(face_df, edge_df, vert_df)

    if area:
        # Approximate area
        edge_df["sub_area"] = (
            np.linalg.norm(edge_df[['nx', 'ny', 'nz']], axis=1) / 2
        )
        face_df['area_approximate'] = _lvl_sum(
            edge_df, edge_df['sub_area'], 'face')

        # "real" area
        face_df['area'] = 0
        for a in face_df.index:
            if a != -1:
                points = projected_points(face_df, edge_df, points_df, a)[
                    ['x', 'y']].to_numpy()
                hull = ConvexHull(points)
                face_df.loc[a, 'area'] = hull.volume

    if perimeter:

        # Approximate perimeter
        edge_df['length_approximate'] = np.sqrt(
            (edge_df.sx-edge_df.tx)**2+(edge_df.sy-edge_df.ty)**2+(edge_df.sz-edge_df.tz)**2).to_numpy()
        face_df['perimeter_approximate'] = edge_df.groupby('face')[
            'length_approximate'].sum()

        # Real perimeter
        edge_df['length'] = 0
        for e in edge_df.index:
            dist_ = 0
            for i in range(len(points_df[points_df.edge == e])-1):
                dist_ += np.sqrt((points_df[points_df.edge == e].iloc[i+1]['x'] - points_df[points_df.edge == e].iloc[i]['x'])**2 +
                                 (points_df[points_df.edge == e].iloc[i+1]['y'] - points_df[points_df.edge == e].iloc[i]['y'])**2 +
                                 (points_df[points_df.edge == e].iloc[i+1]['z'] - points_df[points_df.edge == e].iloc[i]['z'])**2)
            edge_df.loc[e, 'length'] = dist_

        # face_df['perimeter'] = edge_df.groupby('face')[
        #     'length'].sum()
        # Perimeter with convexhull
        for a in face_df.index:
            if a != -1:
                points = projected_points(face_df, edge_df, points_df, a)[
                    ['x', 'y']].to_numpy()
                hull = ConvexHull(points)
                face_df.loc[a, 'perimeter'] = hull.area

    if nb_neighbor:
        face_df['nb_neighbor'] = edge_df.groupby('face')[('srce')].count()

    if aniso:
        points_df[['fz', 'fy', 'fx']] = face_df.loc[points_df.face.to_numpy()
                                                ][['fz', 'fy', 'fx']].to_numpy()
        for f in face_df.index:
            if f != -1:
                points = points_df[points_df.face == f][['x', 'y', 'z']].to_numpy(
                                ).astype("float") - points_df[points_df.face == f][['fx', 'fy', 'fz']].to_numpy().astype("float")

                # Measure cell anisotropie
                u, s, vh = np.linalg.svd(points)

                svd =  np.concatenate((s, vh[0, :]))

                ocoords = ["orientation" + u for u in list('xyz')]
                face_df.loc[f, ocoords] = svd[3:]

                s.sort()
                s = s[::-1]

                aniso = s[0] / s[1]

                face_df.loc[f, 'major'] = s[0]
                face_df.loc[f, 'minor'] = s[1]
                face_df.loc[f, 'aniso'] = aniso

    if j_orientation:
        for e in edge_df.index:
            points = points_df[points_df.edge == e][['x', 'y', 'z']].to_numpy(
                            ).astype("float") - edge_df.loc[e][['sx','sy','sz']].to_numpy().astype("float")

            # Measure edge orientation according to xy plan
            edge_df['orientation_xy'] = np.arctan2(edge_df.ty-edge_df.sy,
                                                   edge_df.tx-edge_df.sx)

            edge_df['orientation_xz'] = np.arctan2(edge_df.tz-edge_df.sz,
                                                   edge_df.tx-edge_df.sx)

            edge_df['orientation_yz'] = np.arctan2(edge_df.tz-edge_df.sz,
                                                   edge_df.ty-edge_df.sy)

            if angle_degree:
                edge_df['orientation_xy'] = edge_df['orientation_xy']*180/np.pi
                edge_df['orientation_xz'] = edge_df['orientation_xz']*180/np.pi
                edge_df['orientation_yz'] = edge_df['orientation_yz']*180/np.pi

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
                       edge_df,
                       points_df,
                       dilation=3,
                       new_column='intensity'
                       ):
    """
    """

    edge_df[new_column+'_mean'] = -1
    edge_df[new_column+'_std'] = -1

    for e, val in edge_df.iterrows():
        x_ = list(points_df[points_df.edge == e]['x_pix'].to_numpy())
        y_ = list(points_df[points_df.edge == e]['y_pix'].to_numpy())
        z_ = list(points_df[points_df.edge == e]['z_pix'].to_numpy())

        image_tmp = np.zeros(image.shape)

        image_tmp[z_, y_, x_] = 1

        s = sci.ndimage.generate_binary_structure(dilation, dilation)
        image_tmp = sci.ndimage.morphology.binary_dilation(
            image_tmp, s).astype(int)

        edge_df.loc[e, new_column +
                    '_mean'] = np.mean(image[np.where(image_tmp == 1)])
        edge_df.loc[e, new_column +
                    '_std'] = np.std(image[np.where(image_tmp == 1)])


def face_intensity(image,
                   face_df,
                   edge_df,
                   vert_df,
                   points_df,
                   thickness,
                   dilation,
                   pixel_size,
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
    image_no_junction[points_df.z_pix.astype(int), points_df.y_pix.astype(int), points_df.x_pix.astype(int)] = 1
    s = sci.ndimage.generate_binary_structure(dilation, dilation)
    image_no_junction = ~sci.ndimage.morphology.binary_dilation(
                                     image_no_junction, s).astype(int)

    image_no_junction = image * image_no_junction
    
    
    all_enlarge_face_id = np.zeros(image.shape)
    update_geom(face_df, edge_df, vert_df)
    compute_normal(face_df, edge_df, vert_df)

    face_df[new_column] = -1
    for f in face_df.index:
        img_face = enlarge_face_plane(image,
                                      face_df,
                                      edge_df,
                                      vert_df,
                                      points_df,
                                      f,
                                      thickness,
                                      pixel_size)
        z, y, x = np.where(img_face>0)
        all_enlarge_face_id[z,y,x] = f+1


        intensity_output = image_no_junction*img_face
        face_df.loc[f, new_column] = np.mean(
            intensity_output[np.where(intensity_output > 0)])
    return all_enlarge_face_id


def enlarge_face_plane(image,
                       face_df,
                       edge_df,
                       vert_df,
                       points_df,
                       face_id,
                       thickness,
                       pixel_size):

    # normal normalisé ? 
    n = (edge_df[edge_df.face == face_id].mean()[['nx', 'ny', 'nz']]/np.linalg.norm(
        edge_df[edge_df.face == face_id].mean()[['nx', 'ny', 'nz']])).to_numpy()
    
    # list points in um   
    xx = points_df[points_df.face==face_id]['x']
    yy = points_df[points_df.face==face_id]['y']
    zz = points_df[points_df.face==face_id]['z']
    
    # Find the top and bottom position according to face plane in um
    top = np.array((xx, yy, zz)).flatten(order='F').reshape((len(xx), 3)) + thickness*n
    bottom = np.array((xx, yy, zz)).flatten(order='F').reshape((len(xx), 3)) - thickness*n
    top = pd.DataFrame(top, columns=[list('xyz')])
    bottom = pd.DataFrame(bottom, columns=[list('xyz')])

    # Convert um position in pixel position
    top['x_pix'] = (top['x']/pixel_size['X_SIZE']).astype('int')
    top['y_pix'] = (top['y']/pixel_size['Y_SIZE']).astype('int')
    top['z_pix'] = (top['z']/pixel_size['Z_SIZE']).astype('int')
    bottom['x_pix'] = (bottom['x']/pixel_size['X_SIZE']).astype('int')
    bottom['y_pix'] = (bottom['y']/pixel_size['Y_SIZE']).astype('int')
    bottom['z_pix'] = (bottom['z']/pixel_size['Z_SIZE']).astype('int')

    # Replace value which exceed boundary to image border value
    top['x_pix'] = np.where((top['x_pix']<0), 0, top['x_pix'])
    top['y_pix'] = np.where((top['y_pix']<0), 0, top['y_pix'])
    top['z_pix'] = np.where((top['z_pix']<0), 0, top['z_pix'])
    bottom['x_pix'] = np.where((bottom['x_pix']<0), 0, bottom['x_pix'])
    bottom['y_pix'] = np.where((bottom['y_pix']<0), 0, bottom['y_pix'])
    bottom['z_pix'] = np.where((bottom['z_pix']<0), 0, bottom['z_pix'])
    
    top['x_pix'] = np.where((top['x_pix']>=image.shape[2]), image.shape[2]-1, top['x_pix'])
    top['y_pix'] = np.where((top['y_pix']>=image.shape[1]), image.shape[1]-1, top['y_pix'])
    top['z_pix'] = np.where((top['z_pix']>=image.shape[0]), image.shape[0]-1, top['z_pix'])
    bottom['x_pix'] = np.where((bottom['x_pix']>=image.shape[2]), image.shape[2]-1, bottom['x_pix'])
    bottom['y_pix'] = np.where((bottom['y_pix']>=image.shape[1]), image.shape[1]-1, bottom['y_pix'])
    bottom['z_pix'] = np.where((bottom['z_pix']>=image.shape[0]), image.shape[0]-1, bottom['z_pix'])
    
    
    img_plane = np.zeros(image.shape)
    try:
        # top plane
        for i, data in top.iterrows():
            img_plane[int(data.z_pix), int(data.y_pix), int(data.x_pix)] = 1
        # bottom plane
        for i, data in bottom.iterrows():
            img_plane[int(data.z_pix), int(data.y_pix), int(data.x_pix)] = 1
        # middle plane
        for i, data in vert_df.loc[edge_df[edge_df.face == face_id].srce.to_numpy()].iterrows():
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


