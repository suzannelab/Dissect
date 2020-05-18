import pandas as pd


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
