import os
import numpy as np


def clean_skeleton(skeleton, save=True):
    """
    Function which clean skeleton.
    Means that all isolated filament or critical point will be removed.
    """

    keep_going = True

    while keep_going:
        keep_going = False

        # Mark critical point connected to 1 filament or less
        cp_to_keep = []
        for c in skeleton.crit:
            if c.nfil < 2:
                cp_to_keep.append(False)
            else:
                cp_to_keep.append(True)
        # Remove critical point connected to 1 filament or less
        skeleton.crit[:] = np.array(skeleton.crit)[cp_to_keep]

        # Mark filament connected to only one critical point
        fil_to_keep = []
        for f in skeleton.fil:
            if f.cp1 not in skeleton.crit:
                fil_to_keep.append(False)
                for i in range(len(skeleton.crit)):
                    if f.cp2 == skeleton.crit[i]:
                        keep_going = True
                        skeleton.crit[i].filId.remove(f)
                        skeleton.crit[i].destCritId.remove(f.cp1)
                        break

            elif f.cp2 not in skeleton.crit:
                fil_to_keep.append(False)
                for i in range(len(skeleton.crit)):
                    if f.cp1 == skeleton.crit[i]:
                        keep_going = True
                        skeleton.crit[i].filId.remove(f)
                        skeleton.crit[i].destCritId.remove(f.cp2)
                        break

            else:
                fil_to_keep.append(True)

        # Remove filament
        skeleton.fil[:] = np.array(skeleton.fil)[fil_to_keep]

    if skeleton.isValid():
        if save:
            skeleton.write_vtp(os.path.join(
                skeleton._filename, "_removefil.vtp"))
    else:
        raise nameError('skeleton not valid')
