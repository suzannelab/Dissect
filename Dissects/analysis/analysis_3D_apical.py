import numpy as np
import pandas as pd

from math import atan, atan2, acos
import scipy as sci
from skimage.draw import polygon

from ..utils.utils import pixel_to_um


def update_geom(face_df, edge_df, vert_df):
    edge_df[['sx', 'sy', 'sz']] = vert_df.loc[edge_df.srce,
                                              list('xyz')].to_numpy()
    edge_df[['tx', 'ty', 'tz']] = vert_df.loc[edge_df.trgt,
                                              list('xyz')].to_numpy()

    # straight line length
    edge_df['length'] = np.sqrt(
        (edge_df.sx-edge_df.tx)**2+(edge_df.sy-edge_df.ty)**2+(edge_df.sz-edge_df.tz)**2).to_numpy()

    face_df[['fx', 'fy', 'fz']] = 0
    for i in face_df.index:
        face_df.loc[i]['fx'] = edge_df[edge_df.face == i].mean()['sx']
        face_df.loc[i]['fy'] = edge_df[edge_df.face == i].mean()['sy']
        face_df.loc[i]['fz'] = edge_df[edge_df.face == i].mean()['sz']


def compute_normal(face_df, edge_df, vert_df):
    face_df.loc[-1] = np.zeros((face_df.shape[1]))
    edge_df[['sz', 'sy', 'sx']] = vert_df.loc[edge_df.srce][[
        'x_um', 'y_um', 'z_um']].to_numpy()
    edge_df[['tz', 'ty', 'tx']] = vert_df.loc[edge_df.trgt][[
        'x_um', 'y_um', 'z_um']].to_numpy()

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


def morphology_analysis(face_df, edge_df, vert_df):

    face_df[['perimeter', 'nb_neighbor', 'aniso', 'area',
             'orient_x', 'orient_y', 'orient_z']] = 0.

    # Area
    edge_df['x'] = vert_df.loc[edge_df.trgt]['x'].to_numpy(
    )-vert_df.loc[edge_df.srce]['x'].to_numpy()
    edge_df['y'] = vert_df.loc[edge_df.trgt]['y'].to_numpy(
    )-vert_df.loc[edge_df.srce]['y'].to_numpy()
    edge_df['z'] = vert_df.loc[edge_df.trgt]['z'].to_numpy(
    )-vert_df.loc[edge_df.srce]['z'].to_numpy()
    edge_df["sub_area"] = (
        np.linalg.norm(edge_df[list('xyz')], axis=1) / 2
    )
    face_df['area'] = _lvl_sum(edge_df, edge_df['sub_area'], 'face')

    for f in face_df.index:
        face_df.loc[f]['perimeter'] = edge_df[edge_df.face == f].sum().length
        face_df.loc[f]['nb_neighbor'] = edge_df[edge_df.face == f].shape[0]

    for f in face_df.index:
        points = vert_df.loc[edge_df[edge_df.face == f]['srce']][list(
            'xyz')].to_numpy() - face_df.loc[f][['fx', 'fy', 'fz']].to_numpy()

        # Measure cell anisotropie
        u, s, vh = np.linalg.svd(points)
        # Euler angle
        vh = vh.T
        sy = np.sqrt(vh[0, 0] * vh[0, 0] + vh[1, 1]
                     * vh[1, 1] + vh[2, 2] * vh[2, 2])

        singular = sy < 1e-6
        if not singular:
            psi = np.abs(atan2(vh[2, 1], vh[2, 2]))
            theta = np.abs(atan2(vh[2, 0], sy))
            phi = np.abs(atan2(vh[1, 0], vh[0, 0]))
        else:
            psi = atan2(-vh[1, 2], vh[1, 1])
            theta = atan2(-vh[2, 0], sy)
            phi = 0
        theta = np.abs(
            acos(vh[2, 0] / np.sqrt(vh[0, 0]**2 + vh[1, 0]**2 + vh[2, 0]**2)))
        phi = np.abs(atan(vh[1, 0] / vh[0, 0]))

        s.sort()
        s = s[::-1]

        aniso = s[0] / s[1]
        orientation = aniso * np.array([np.sin(theta) * np.cos(phi),
                                        np.sin(theta) * np.sin(phi),
                                        np.cos(theta)])

        face_df.loc[f]['orient_x'] = orientation[0]
        face_df.loc[f]['orient_y'] = orientation[1]
        face_df.loc[f]['orient_z'] = orientation[2]
        face_df.loc[f]['aniso'] = aniso

# sum_face(edge_df["sub_area"])


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
                        new_column='intensity'):

    edge_df[new_column] = -1
    for e, val in edge_df.iterrows():
        try:
            z_ = np.fromstring(
                val.points[0][1:-1].split('[')[1].split(']')[0], sep=',').astype('int')
            y_ = np.fromstring(
                val.points[0][1:-1].split('[')[2].split(']')[0], sep=',').astype('int')
            x_ = np.fromstring(
                val.points[0][1:-1].split('[')[3].split(']')[0], sep=',').astype('int')

            image_tmp = np.zeros(image.shape)

            image_tmp[z_, y_, x_] = 1
            image_tmp = sci.ndimage.morphology.binary_dilation(image_tmp,
                                                   sci.ndimage.generate_binary_structure(3, 3)).astype(int)
            edge_df.loc[e, new_column] = np.mean(
                image[np.where(image_tmp == 1)])
        except Exception as ex:
            """
            correspond aux petites jonction ajouté "à la main"
            il faut retrouver les pixels pour pouvoir mesurer les jonctions...
            """
            print(e)
            print(ex)
            pass


def face_intensity(image,
                   face_df,
                   edge_df,
                   vert_df,
                   thicken,
                   pixel_size,
                   new_column='intensity'):
    face_df[new_column] = -1
    for f in face_df.index:
        img_face = enlarge_face_plane(image,
                                      face_df,
                                      edge_df,
                                      vert_df,
                                      f,
                                      thicken,
                                      pixel_size)
        intensity_output = image*img_face
        face_df.loc[f, new_column] = np.mean(
            intensity_output[np.where(intensity_output > 0)])


def enlarge_face_plane(image,
                       face_df,
                       edge_df,
                       vert_df,
                       face_id,
                       thicken,
                       pixel_size):

    pixel_to_um(vert_df, pixel_size, list('xyz'), ['x_um', 'y_um', 'z_um'])
    compute_normal(face_df, edge_df, vert_df)

    n = (edge_df[edge_df.face == face_id].mean()[['nx', 'ny', 'nz']]/np.linalg.norm(
        edge_df[edge_df.face == face_id].mean()[['nx', 'ny', 'nz']])).to_numpy()
    # list points
    zz = np.empty(0)
    yy = np.empty(0)
    xx = np.empty(0)
    for data in edge_df[edge_df.face == face_id].points:
        try:
            zz = np.concatenate(
                (zz, np.fromstring(data[0].split('[')[1].split(']')[0], sep=',')))
            yy = np.concatenate(
                (yy, np.fromstring(data[0].split('[')[2].split(']')[0], sep=',')))
            xx = np.concatenate(
                (xx, np.fromstring(data[0].split('[')[3].split(']')[0], sep=',')))
        except:
            pass

    xx = np.concatenate(
        (xx, vert_df.loc[edge_df[edge_df.face == face_id].srce.to_numpy()]['x'].to_numpy()))
    yy = np.concatenate(
        (yy, vert_df.loc[edge_df[edge_df.face == face_id].srce.to_numpy()]['y'].to_numpy()))
    zz = np.concatenate(
        (zz, vert_df.loc[edge_df[edge_df.face == face_id].srce.to_numpy()]['z'].to_numpy()))

    top = np.array((xx*pixel_size['x'], yy*pixel_size['y'], zz*pixel_size['z'])
                   ).flatten(order='F').reshape((len(xx), 3))+thicken*n
    bottom = np.array((xx*pixel_size['x'], yy*pixel_size['y'], zz *
                       pixel_size['z'])).flatten(order='F').reshape((len(xx), 3))-thicken*n
    top = pd.DataFrame(top, columns=['x_um', 'y_um', 'z_um'])
    bottom = pd.DataFrame(bottom, columns=['x_um', 'y_um', 'z_um'])

    top['x'] = round(top['x_um']/pixel_size['x'])
    top['y'] = round(top['y_um']/pixel_size['y'])
    top['z'] = round(top['z_um']/pixel_size['z'])

    bottom['x'] = round(bottom['x_um']/pixel_size['x'])
    bottom['y'] = round(bottom['y_um']/pixel_size['y'])
    bottom['z'] = round(bottom['z_um']/pixel_size['z'])

    try:
        img_plane = np.zeros(image.shape)
        for i, data in top.iterrows():
            img_plane[int(data.z), int(data.y), int(data.x)] = 1
        for i, data in bottom.iterrows():
            img_plane[int(data.z), int(data.y), int(data.x)] = 1
        for i, data in vert_df.loc[edge_df[edge_df.face == face_id].srce.to_numpy()].iterrows():
            img_plane[int(data.z), int(data.y), int(data.x)] = 1
    except Exception as ex:
            """
            correspond aux petites jonction ajouté "à la main"
            il faut retrouver les pixels pour pouvoir mesurer les jonctions...
            """
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

    # enlever une rangée de pixel autour

    return img_plane
