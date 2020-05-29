import pandas as pd
import numpy as np


def create_df_from_skeleton(skeleton):
    data_crit = []

    for c in skeleton.crit:
        data_crit.append(
            {
                "id": c,
                "x": c.pos[0],
                "y": c.pos[1],
                "z": c.pos[2] * 0.22 / 0.18,
                "n_fil": c.nfil,
                "val": int(c.val),
                "pair": c.pair,
                "type": int(c.typ),
                "destCritId_addr": c.destCritId,
                "filId": c.filId,
            }
        )

    data_crit = pd.DataFrame.from_records(data_crit)
    tmp_id = []
    for ci in data_crit['destCritId_addr'].values:
        tmp = []
        for c in ci:
            tmp.append(data_crit[data_crit['id'] == c].index.to_numpy()[0])

        tmp_id.append(tmp)

    data_crit['destCritId'] = tmp_id

    data_fil = []

    for f in skeleton.fil:
        data_fil.append(
            {
                "id": f,
                "cp1_add": f.cp1,
                "cp2_add": f.cp2,
                "cp1": data_crit[data_crit['id'] == f.cp1].index.to_numpy()[0],
                "cp2": data_crit[data_crit['id'] == f.cp2].index.to_numpy()[0],
                "points": f.points

            }
        )

    data_fil = pd.DataFrame.from_records(data_fil)

    return data_crit, data_fil


def fil_dilation(skeleton, img_original, width):
    # Peut etre que l'on peut utiliser une matrice de dilatation
    # comme scipy.binary_dilation (fonctionne qu'en 2D?...)

    # A modifier pour se passer de passer en paramètre
    # l'image originale pour récupérer les dimensions de l'image

    img_dilate = img_original.copy()
    img_dilate = np.where(img_dilate > 0, 0, img_dilate)
    for f in skeleton.fil:
        for p in f.points:
            if width != 0:
                for z_ in range(int(p[2]) - width, int(p[2]) + width):
                    for y_ in range(int(p[1]) - width, int(p[1]) + width):
                        for x_ in range(int(p[0]) - width, int(p[0]) + width):
                            try:
                                img_dilate[z_][y_][x_] = 1
                            except:
                                pass
            else:
                x_ = int(p[0])
                y_ = int(p[1])
                z_ = int(p[2])
                img_dilate[z_][y_][x_] = 1
    return img_dilate


def img_dilation(img, width=2):
    if width == 0:
        return img
    img_dilate = img.copy()

    z, y, x = np.where(img > 0)
    for i in range(len(z)):
        for z_ in range(z[i] - width, z[i] + width):
            for y_ in range(y[i] - width, y[i] + width):
                for x_ in range(x[i] - width, x[i] + width):
                    try:
                        img_dilate[z_][y_][x_] = 1
                    except:
                        pass
    return img_dilate


def xyz_from_array(img_array):
    lx = []
    ly = []
    lz = []
    for z in range(img_array.shape[0]):
        for y in range(img_array.shape[1]):
            for x in range(img_array.shape[2]):
                if img_array[z][y][x] != 0:
                    lx.append(x)
                    ly.append(y)
                    lz.append(z)
    return lx, ly, lz


def filmask_int(skeleton, im):
    """
    Assign integer position to segment's points.

    If the coordinate is higher than image size,
    the position is brought to the edge.
    """
    mask = np.zeros_like(im)
    for i in range(skeleton.nfil):
        for j in range(len(skeleton.fil[i].points)):
            ii = skeleton.fil[i].points[j, 1].astype(int)
            if ii < 0:
                ii = 0
            if ii >= mask.shape[0]:
                ii = mask.shape[0] - 1
            jj = skeleton.fil[i].points[j, 0].astype(int)
            if jj < 0:
                jj = 0
            if jj >= mask.shape[1]:
                jj = mask.shape[1] - 1
            mask[ii, jj] = 1
    return mask
