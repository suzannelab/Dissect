from base import CriticalPoint, Filament
from scipy.spatial import cKDTree as KDTree
from collections import deque
from tvtk.api import tvtk
import matplotlib.pyplot as plt
import numpy as np
import itertools
import mpu


class SkelError(Exception):
    pass


# read a line, skipping comment lines
def _readline(f):
    _COMMENT = '#'
    for line in f:
        if line.find(_COMMENT) != 0:
            break
    return line


def _check_p(f, pattern, optional=False):
    line = _readline(f)
    if line.find(pattern) != 0:
        if not(optional):
            raise SkelError('wrong format, missing: {0}'.format(pattern))
    else:
        return line


class Skel():

    def __init__(self, filename=''):
        self.crit = []  # create a list of critical points
        self.fil = []
        if filename:
            self._filename = filename
            self.read_NDskl_ascii()
            self._chain()
            self.isValid()

    @property
    def nfil(self):
        return len(self.fil)

    @property
    def ncrit(self):
        return len(self.crit)

    def read_NDskl_ascii(self):
        with open(self._filename, 'r') as f:
            if _check_p(f, 'ANDSKEL'):
                self.ndims = int(_readline(f))

            line = _check_p(f, 'BBOX', optional=True)
            if line:
                start = line.find('BBOX') + 4
                s1 = line.find('[', start)
                s2 = line.find(']', start)
                sub = line[s1 + 1:s2]
                self.bbox = np.asfarray(sub.split(','))
                start = s2 + 1
                s1 = line.find('[', start)
                s2 = line.find(']', start)
                sub = line[s1 + 1:s2]
                self.bbox_delta = np.asfarray(sub.split(','))

            if _check_p(f, '[CRITICAL POINTS]'):
                ncrit = int(_readline(f))
                print('reading: {0} critical points'.format(ncrit))
                for _ in range(ncrit):

                    # read 1st line: info on the cp
                    data = _readline(f).split()
                    typ = int(data[0])
                    pos = np.array([np.float32(x)
                                    for x in data[1:1 + self.ndims]])
                    val = float(data[1 + self.ndims])
                    pair = int(data[2 + self.ndims])
                    boundary = int(data[3 + self.ndims])

                    # read 2nd line:  number of filaments that connect to the
                    # CP
                    nfil = int(_readline(f))

                    # read nfil lines: destination and cp of the nfil filaments
                    destCritId = []
                    filId = []
                    for _ in range(nfil):
                        data = _readline(f).split()
                        destCritId.append(int(data[0]))
                        filId.append(int(data[1]))

                    this_crit = CriticalPoint(typ, pos, val, pair,
                                              boundary, destCritId, filId)
                    self.crit.append(this_crit)

            if _check_p(f, '[FILAMENTS]'):
                nfil = int(_readline(f))
                print('reading: {0} filaments'.format(nfil))
                for _ in range(nfil):
                    #print('reading filament i:{0}'.format(i))
                    data = _readline(f).split()
                    cp1 = int(data[0])
                    cp2 = int(data[1])
                    nsamp = int(data[2])
                    fil_points = np.zeros([nsamp, self.ndims], dtype='float32')
                    # read points
                    n_to_read = nsamp * self.ndims
                    index = 0
                    while n_to_read:
                        data = np.asfarray(_readline(f).split())
                        npoints = np.size(data) // self.ndims
                        fil_points[index:index + npoints, :] = \
                            data.reshape([npoints, self.ndims])
                        index += npoints
                        n_to_read -= np.size(data)
                    this_fil = Filament(
                        self.crit[cp1], self.crit[cp2], fil_points)
                    self.fil.append(this_fil)

            if _check_p(f, '[CRITICAL POINTS DATA]', optional=True):
                print('reading: critical points data')
                # read data names
                self.ncrit_data = int(_readline(f))
                self.crit_data_name = []
                for _ in range(self.ncrit_data):
                    self.crit_data_name.append(_readline(f).strip())
                # read data
                for cp in self.crit:
                    cp.data = [eval(Str) for Str in _readline(f).split()]
                print('critical points done')

            if _check_p(f, '[FILAMENTS DATA]', optional=True):
                print('reading: filaments data')
                # read data names
                self.nfil_data = int(_readline(f))
                self.fil_data_name = []
                for _ in range(self.nfil_data):
                    self.fil_data_name.append(_readline(f).strip())
                # read data
                for fil in self.fil:
                    fil_data = np.zeros(
                        [fil.nsamp, self.nfil_data], dtype='float32')
                    n_to_read = fil.nsamp * self.nfil_data
                    index = 0
                    while n_to_read:
                        data = np.asfarray(_readline(f).split())
                        npoints = np.size(data) // self.nfil_data
                        fil_data[index:index + npoints, :] = \
                            data.reshape([npoints, self.nfil_data])
                        index += npoints
                        n_to_read -= np.size(data)
                    fil.data = fil_data

    # replace id numbers pointing to filament
    # or critical points by there object reference
    def _chain(self):
        for cp in self.crit:
            cp.pair = self.crit[cp.pair]
            cp.destCritId = [self.crit[i] for i in cp.destCritId]
            cp.filId = [self.fil[i] for i in cp.filId]

    def isValid(self):
        """check the consistency of the tree"""
        for crit in self.crit:
            for j in range(crit.nfil):
                fil = crit.filId[j]
                if (fil.cp1 == crit and fil.cp2 != crit.destCritId[j]):
                    raise SkelError('wrong critical point {0} and associated filament {1}'
                                    .format(self.crit.index(crit), j))
                elif (fil.cp2 == crit and fil.cp1 != crit.destCritId[j]):
                    raise SkelError('wrong critical point {0} and associated filament {1}'
                                    .format(self.crit.index(crit), j))
                elif(fil.cp1 != crit and fil.cp2 != crit):
                    raise SkelError('wrong critical point {0} and associated filament {1}'
                                    .format(self.crit.index(crit), j))
        bad = good = []
        for i, fil in enumerate(self.fil):
            if not(fil in fil.cp1.filId) or not(fil in fil.cp2.filId):
                # don't raise the error but remove the filament from the list
                print("wrong filament {0}, \
                    not listed in its critical points".format(i))
                bad.append(fil)
                # raise SkelError("wrong filament {0}, \
                #   not listed in its critical points".format(i))
            else:
                good.append(fil)

        if bad:
            self.fil[:] = good
        return True

# 3D SKELETON
    @property
    def is_broken_down(self):
        """Check if the breakdown option was set when running mse
        """
        try:
            return self._is_broken_down
        except AttributeError:
            typ4 = [c.typ for c in self.crit if c.typ == 4]
            if len(typ4) == 0:
                self._is_broken_down = False
            else:
                self._is_broken_down = True
            return self._is_broken_down

    def filter_nodes_alone(self, filter_filaments=True):
        """remove the nodes with only one filament connected to it
            and eventually removes the associated filaments
        """
        #alones_idx, alones_cp =zip(*[(i, cp) for i, cp in enumerate(self.crit) if cp.nfil==1 and cp.typ==3])
        alones_idx = [i for i, cp in enumerate(
            self.crit) if cp.nfil == 1 and cp.typ == 3]

        # mark bad maxima to be removed
        mask = np.ones(self.ncrit, dtype=bool)
        mask[alones_idx] = False

        if filter_filaments:
            mask_fil = np.ones(self.nfil, dtype=bool)

        for i in alones_idx:
            cp = self.crit[i]
            # remove persistence pair reference
            cp.pair.pair = None

            # remove the connected filament
            if filter_filaments:
                # mark the filament to be removed
                mask_fil[self.fil.index(cp.filId[0])] = False
                # remove the connection in the saddle
                saddle = cp.destCritId[0]
                Id_in_saddle = saddle.destCritId.index(cp)
                saddle.unconnect_fil(Id_in_saddle)
                # if saddle is alone remove it
                if saddle.nfil == 0:
                    mask[self.crit.index(saddle)] = False

        # remove the maxima
        self.crit[:] = np.array(self.crit)[mask]

        # remove all marked filaments
        if filter_filaments:
            self.fil[:] = np.array(self.fil)[mask_fil]

    def filter_spurious_saddles(self):
        """remove the spurious saddles on the border superposed to maxima
        plus the filament of null-length that connect them.
        """
        mask = np.ones(self.ncrit, dtype=bool)
        mask_fil = np.ones(self.nfil, dtype=bool)
        for i, f in enumerate(self.fil):
            if not np.linalg.norm(f.cp1.pos - f.cp2.pos):
                mask_fil[i] = False
                saddle = f.cp1
                saddleId = self.crit.index(saddle)
                #del self.crit[saddleId]
                mask[saddleId] = False
                # remove persistence pair ref
                f.cp1.pair.pair = None
                filId_in_max = f.cp2.filId.index(f)
                f.cp2.unconnect_fil(filId_in_max)
                # if saddle is connected to another max, remove the connection
                if f.cp1.nfil == 2:
                    f2id = 1
                    if saddle.filId[0] != f:
                        f2id = 0
                    f2 = saddle.filId[f2id]
                    mask_fil[self.fil.index(f2)] = False
                    f2.cp1.pair.pair = None
                    filId_in_max = f2.cp2.filId.index(f2)
                    f2.cp2.unconnect_fil(filId_in_max)

        self.fil[:] = np.array(self.fil)[mask_fil]
        self.crit[:] = np.array(self.crit)[mask]

    def generate_Ids(self):
        self.ncrit_data += 1
        self.crit_data_name.append('OriginalId')
        for i, cp in enumerate(self.crit):
            cp.data.append(i)

    def distance_to_nearest_node(self, points):
        """compute the distance of a given point to
        nearest node (critical point 3)"""
        # nodes = (x for x in self.crit if x.typ==3)
        try:
            tree = self._node_tree
        except AttributeError:
            nodes_pos = np.array([x.pos for x in self.crit if x.typ == 3])
            self._node_tree = KDTree(nodes_pos)
            tree = self._node_tree
        d, idx = tree.query(points)
        crits_id = np.array([i for i, x in enumerate(self.crit) if x.typ == 3])
        return d, crits_id[idx]

    def distance_to_node(self, crit_index, points):
        """compute the distance of a given point to
        a given node (critical point 3)"""
        # nodes = (x for x in self.crit if x.typ==3)
        p = self.crit[crit_index]
        if p.typ != 3:
            raise SkelError('wrong type, max point expected (type 3)')
        d = np.sqrt((p.pos[0] - points[0])**2 + (p.pos[1] -
                                                 points[1])**2 + (p.pos[2] - points[2])**2)
        return d

    def distance_to_nearest_saddle(self, points):
        """compute the distance of a given point to
        nearest saddle-2 (critical point 2)"""
        # nodes = (x for x in self.crit if x.typ==3)
        try:
            tree = self._saddle_tree
        except AttributeError:
            saddles_pos = np.array([x.pos for x in self.crit if x.typ == 2])
            tree = self._saddle_tree = KDTree(saddles_pos)
        d, idx = tree.query(points)
        saddles_id = np.array(
            [i for i, x in enumerate(self.crit) if x.typ == 2])
        return d, saddles_id[idx]

    def distance_to_skel(self, points, big=None):
        """compute the distance of a given point to the nearest filament"""
        try:
            b = self._big_segments_limit
        except AttributeError:
            pass
        else:
            if b != big:
                del self._seg_tree

        try:
            tree = self._seg_tree
        except AttributeError:
            segs_pos = np.vstack([x.mid_segments() for x in self.fil])
            self._big_segments_limit = big
            if big:
                # remove big segments (expect boundary removal)
                segs_len = np.hstack([x.segments_len for x in self.fil])
                segs_keep, = (segs_len < big).nonzero()
                segs_pos = segs_pos[segs_keep]
            # print 'number of segments:',len(segs_pos)
            self._seg_tree = KDTree(segs_pos)
            tree = self._seg_tree

        distances, indexes = tree.query(points)
        if big:
            indexes = segs_keep[indexes]

        if not self.is_broken_down:
            # deals with overlapped filaments (no breakdown case)
            points = np.array(points)
            indexes = [tree.query_ball_point(
                points[i], distances[i] + 1e-7) for i in range(points.shape[0])]
            n_grp = np.array(list(map(len, indexes)))
            start_grp = np.concatenate(([0], np.cumsum(n_grp[:-1])))
            indexes = list(itertools.chain.from_iterable(indexes))  # flat list

        # get the filament number
        istart_fil = np.cumsum(np.array([x.nsamp - 1 for x in self.fil[:-1]]))
        istart_fil = np.concatenate(([0], istart_fil))
        fil_indexes = np.searchsorted(istart_fil, indexes, side='right') - 1
        seg_indexes = indexes - istart_fil[fil_indexes]

        if not self.is_broken_down:
            # regroup the flat lists
            fil_indexes = [fil_indexes[s:s + n]
                           for s, n in zip(start_grp, n_grp)]
            seg_indexes = [seg_indexes[s:s + n]
                           for s, n in zip(start_grp, n_grp)]

        return distances, fil_indexes, seg_indexes

    def distance_along_filament_to_node(self, fil_index, seg_index):
        '''Compute the distance inside a filament (starting at seg_index) up to node.
        For nodes there is no ambiguity: going up certainly lead to one unique node
        '''
        f = self.fil[fil_index]
        fil_lst, cp = self.follow_filament_to_cp(f.cp2, f)
        if fil_lst is None:
            raise SkelError("could not follow filament to node")
        f_l = np.array(fil_lst).flatten()
        d = f.segments_cumlen_from_cp2[seg_index]
        i = 1
        while i < len(f_l):
            f = f_l[i]
            d += f.len  # segments_cumlen_from_cp2[0]
            i += 1

        if type(cp) == list:
            c = cp[0]
            while type(c) != CriticalPoint:
                c = c[0]
            cp_id = self.crit.index(c)
        else:
            cp_id = self.crit.index(cp)
        return d, cp_id

    def distance_along_filament_to_saddle(self, fil_index, seg_index):
        '''Compute the distance inside a filament (starting at seg_index) up to saddle.
        This implementation is only valid for non broken skeleton (ie it leads to a unique saddle)
        '''
        f = self.fil[fil_index]
        fil_lst, cp = self.follow_filament_to_cp(f.cp1, f, node=False)
        if fil_lst is None:
            raise SkelError("could not follow filament to saddle")
        d = f.segments_cumlen_from_cp1[seg_index]
        fil_lst.popleft()
        while len(fil_lst):
            d += fil_lst.popleft().len
        cp_id = self.crit.index(cp)
        return d, cp_id

    def filaments_from_saddle(self, crit_index):
        p = self.crit[crit_index]
        if p.typ != 2:
            raise SkelError('wrong type, saddle point expected (type 2)')
        all_fil_list = []
        all_node_list = []
        for i in range(p.nfil):
            #print('tracing filament {0}/{1}'.format(i,p.nfil))
            fil = p.filId[i]
            dest = p.destCritId[i]
            res, node = self.follow_filament_to_cp(dest, fil)
            if res != None:
                all_fil_list.append(res)
                all_node_list.append(node)  # only one is returned
        return all_fil_list, all_node_list

    def filaments_from_node(self, crit_index):
        '''return the list of filaments and saddles connected to node
        This routine is useful in the case of broken skeleton (option -breakdown of skelconv)
        otherwise it is straightforward.
        Each filament is returned as a list of broken filament
        (ie with bifurcation point at extremities)
        '''
        p = self.crit[crit_index]
        if p.typ != 3:
            raise SkelError('wrong type, max point expected (type 3)')
        all_fil_list = []
        all_saddle_list = []
        for i in range(p.nfil):
            fil = p.filId[i]
            dest = p.destCritId[i]
            res, saddle = self.follow_filament_to_cp(dest, fil, node=False)
            if res != None:
                all_fil_list.append(res)
                all_saddle_list.append(saddle)  # several may be returned
        return all_fil_list, all_saddle_list

    def follow_filament_to_cp(self, p, fil, node=True):
        """follow filament chain through bifurcation points up to node
           (if node=True) or up to saddle (if node=False)
        """
        if node:
            cpgood = self.ndims
            cpbad = self.ndims - 1
        else:
            cpgood = self.ndims - 1
            cpbad = self.ndims
        if p.typ == cpgood:
            return deque([fil]), p
        if p.typ == cpbad:
            return None, None
        # then bifurcation
        new_fil_lst = None
        new_cp_lst = None
        for i in range(p.nfil):
            f = p.filId[i]
            if f == fil:
                continue
            this_p = p.destCritId[i]
            fil_lst, cp_lst = self.follow_filament_to_cp(this_p, f, node)
            if fil_lst != None:
                if all([isinstance(x, Filament) for x in fil_lst]):
                    fil_lst.appendleft(fil)
                else:
                    for c in fil_lst:
                        c.appendleft(fil)
                if new_fil_lst is None:
                    new_fil_lst = deque([fil_lst])
                    new_cp_lst = [cp_lst]
                else:
                    new_fil_lst.append(fil_lst)
                    if new_cp_lst is None:
                        new_cp_lst = [cp_lst]
                    else:
                        new_cp_lst.extend([cp_lst])
                if len(new_fil_lst) == 1:
                    new_fil_lst = deque([new_fil_lst[0]])
                    new_cp_lst = [new_cp_lst[0]]
        return new_fil_lst, new_cp_lst

    def connectivity3D(self, crit_index, R):
        """computes connectivity as the number of filaments crossing 1.5 x R200"""
        p = self.crit[crit_index]
        if p.typ != 3:
            raise SkelError('wrong type, max point expected (type 3)')
        fil_lst = np.array(self.filaments_from_node(crit_index)[0]).flatten()
        c3d = 0
        for fil in fil_lst:
            ctmp = 0
            l = 0
            while ctmp < 1 and l < len(fil.points[:-1]):
                p1 = fil.points[l]
                p2 = fil.points[l + 1]
                d1 = np.sqrt((p.pos[0] - p1[0])**2 +
                             (p.pos[1] - p1[1])**2 + (p.pos[2] - p1[2])**2)
                d2 = np.sqrt((p.pos[0] - p2[0])**2 +
                             (p.pos[1] - p2[1])**2 + (p.pos[2] - p2[2])**2)
                if d1 < R and d2 > R:
                    c3d += 1
                    ctmp += 1
                if d2 < R and d1 > R:
                    c3d += 1
                    ctmp += 1
                l = l + 1
        return c3d

    def fof_arround_max(self, delaunay_cat, fieldname, densfrac=.1, fof_max=30):
        """compute fof,
                starting from max,
                stopping at the density fraction densfrac between max and the highest connected saddle.
                (densfrac=0 means stop at the density of the saddle)
        """
        print("Computing fof arround max")
        typ3Id = np.array([i for i, x in enumerate(self.crit) if x.typ == 3])
        tree = KDTree(delaunay_cat.points)
        pos3 = np.array([self.crit[i].pos for i in typ3Id])
        _, nearest_id = tree.query(pos3)
        field_value = delaunay_cat.point_data.get_array(fieldname).to_array()
        fof_indices = np.empty(delaunay_cat.number_of_points, dtype=int)
        fof_indices.fill(-1)
        fof_size = np.empty(typ3Id.size, dtype=int)
        fof_size.fill(-1)
        for i, Id in enumerate(typ3Id):
            cells = delaunay_cat.get_cells()
            # print "\ncrit point {0}".format(i)
            cp = self.crit[Id]
            # get all the connected saddle and set the threshold to the highest
            _, saddle_lst = self.filaments_from_node(Id)
            if not saddle_lst:
                continue  # no filaments from this node...
            saddle_dens = np.array(
                [x.val for x in np.flatten(saddle_lst)])
            density_thres = saddle_dens.max()
            density_thres += densfrac * (cp.val - density_thres)
            # follow delaunay graph for nearest points above thres
            #####################################################
            # start the point set with nearest neighbor of node
            point_set = np.array([nearest_id[i]])
            curpoint = 0
            while curpoint < point_set.size and point_set.size < fof_max:
                # print curpoint,'/',point_set.size
                # find cells containing current point
                cellsId = (np.where(cells == point_set[curpoint]))[0]
                # add in the set the points of the selected cells and above
                # density threshold
                if cellsId.size != 0:
                    points = np.unique(np.take(cells, cellsId, axis=0))
                    points_val = np.take(field_value, points)
                    # & (points_val < cp.val)
                    points = np.compress(points_val > density_thres, points)
                    point_new = np.setdiff1d(
                        points, point_set, assume_unique=True)
                    point_set = np.append(point_set, point_new)
                    np.delete(cells, cellsId, axis=0)
                curpoint += 1
            if point_set.size >= fof_max:
                print("crit point {0}".format(i), "fof too big, stopping...")
            else:
                fof_indices[point_set] = i
                fof_size[i] = point_set.size
            # print "fof size: ", point_set.size
        return fof_indices, fof_size

    def convert_distance(self, conversion):
        for fil in self.fil:
            fil.convert_distance(conversion)
        for crit in self.crit:
            crit.convert_distance(conversion)

    @property
    def len(self):
        return np.sum([f.len for f in self.fil])

    @property
    def mean_len(self):
        return np.mean([f.len for f in self.fil])

    def persistence_diagram(self):
        persistence_ratio_id = self.crit_data_name.index('persistence_ratio')
        persistence_ratio = np.array(
            [x.data[persistence_ratio_id] for x in self.crit if x.typ <= 3])

        persistence_nsig_id = self.crit_data_name.index('persistence_nsigmas')
        persistence_nsig = np.array(
            [x.data[persistence_nsig_id] for x in self.crit if x.typ <= 3])

        persistence_id = self.crit_data_name.index('persistence')
        persistence = np.array([x.data[persistence_id]
                                for x in self.crit if x.typ <= 3])

        ppair_id = self.crit_data_name.index('persistence_pair')
        ppair = np.array([x.data[ppair_id] for x in self.crit if x.typ <= 3])

        field_val_id = self.crit_data_name.index('field_value')
        field_val = np.array([x.data[field_val_id]
                              for x in self.crit if x.typ <= 3])

        typ = np.array([x.typ for x in self.crit if x.typ <= 3])

        ppair_field_val = field_val[ppair]

        good = (np.where(persistence_ratio != -1))[0]
        low = (np.where(ppair_field_val[good] > field_val[good]))[0]
        pair0 = (np.where(typ[good[low]] == 0))[0]
        pair1 = (np.where(typ[good[low]] == 1))[0]
        pair2 = (np.where(typ[good[low]] == 2))[0]

        plt.figure()
        plt.loglog(field_val[good[low[pair0]]], persistence_nsig[good[low[pair0]]],
                   '.', label='min-saddle1')
        plt.loglog(field_val[good[low[pair1]]], persistence_nsig[good[low[pair1]]],
                   '.', label='saddle1-saddle2')
        plt.loglog(field_val[good[low[pair2]]], persistence_nsig[good[low[pair2]]],
                   '.', label='saddle2-max')
        plt.xlabel('density')
        plt.ylabel('persistence nsigmas')
        plt.legend()

        plt.figure()
        plt.loglog(field_val[good[low[pair0]]], persistence_ratio[good[low[pair0]]],
                   '.', label='min-saddle1', alpha=0.2)
        plt.loglog(field_val[good[low[pair1]]], persistence_ratio[good[low[pair1]]],
                   '.', label='saddle1-saddle2', alpha=0.2)
        plt.loglog(field_val[good[low[pair2]]], persistence_ratio[good[low[pair2]]],
                   '.', label='saddle2-max', alpha=0.2)
        plt.xlabel('density')
        plt.ylabel('persistence ratio')
        plt.legend()

        from scipy import stats
        fig, ax = plt.subplots(1, 3)

        xmin0 = np.min(np.log10(field_val[good[low[pair0]]]))
        ymin0 = np.min(np.log10(persistence_ratio[good[low[pair0]]]))
        xmax0 = np.max(np.log10(field_val[good[low[pair0]]]))
        ymax0 = np.max(np.log10(persistence_ratio[good[low[pair0]]]))
        xmin1 = np.min(np.log10(field_val[good[low[pair1]]]))
        ymin1 = np.min(np.log10(persistence_ratio[good[low[pair1]]]))
        xmax1 = np.max(np.log10(field_val[good[low[pair1]]]))
        ymax1 = np.max(np.log10(persistence_ratio[good[low[pair1]]]))
        xmin2 = np.min(np.log10(field_val[good[low[pair2]]]))
        ymin2 = np.min(np.log10(persistence_ratio[good[low[pair2]]]))
        xmax2 = np.max(np.log10(field_val[good[low[pair2]]]))
        ymax2 = np.max(np.log10(persistence_ratio[good[low[pair2]]]))

        xmin = np.min([xmin0, xmin1, xmin2])
        ymin = np.min([ymin0, ymin1, ymin2])
        xmax = np.max([xmax0, xmax1, xmax2])
        ymax = np.max([ymax0, ymax1, ymax2])

        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([np.log10(field_val[good[low[pair0]]]), np.log10(
            persistence_ratio[good[low[pair0]]])])
        kernel = stats.gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, X.shape)
        ax[0].imshow(np.rot90(Z), cmap=plt.cm.Blues,
                     extent=[xmin, xmax, ymin, ymax])

        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([np.log10(field_val[good[low[pair1]]]), np.log10(
            persistence_ratio[good[low[pair1]]])])
        kernel = stats.gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, X.shape)
        ax[1].imshow(np.rot90(Z), cmap=plt.cm.Greens,
                     extent=[xmin, xmax, ymin, ymax])

        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([np.log10(field_val[good[low[pair2]]]), np.log10(
            persistence_ratio[good[low[pair2]]])])
        kernel = stats.gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, X.shape)
        ax[2].imshow(np.rot90(Z), cmap=plt.cm.Reds,
                     extent=[xmin, xmax, ymin, ymax])
        fig.xlabel('log_density')
        fig.ylabel('log_persistence ratio')

        plt.figure()
        plt.loglog(field_val[good[low[pair0]]], persistence[good[low[pair0]]],
                   '.', label='min-saddle1')
        plt.loglog(field_val[good[low[pair1]]], persistence[good[low[pair1]]],
                   '.', label='saddle1-saddle2')
        plt.loglog(field_val[good[low[pair2]]], persistence[good[low[pair2]]],
                   '.', label='saddle2-max')
        plt.xlabel('density')
        plt.ylabel('persistence')
        plt.legend(loc=4)

        plt.show()

    def write_vtp(self, filename):
        """
        write skeleton to a vtk PolyData file (.vtp format)
        """
        # get nb o f  points
        npoints = self.ncrit
        fil_npoints = np.array([fil.nsamp - 2 for fil in self.fil])
        fil_npoints = fil_npoints.sum()
        npoints += fil_npoints

        points = np.zeros((npoints, 3))
        verts = np.arange(self.ncrit)[:, np.newaxis]
        lines = []
        for i, crit in enumerate(self.crit):
            points[i] = crit.pos
        start = self.ncrit
        for fil in self.fil:
            end = start + fil.nsamp - 2
            points[start:end] = fil.points[1:-1]
            line = [self.crit.index(fil.cp1)]
            line.extend(list(range(start, end)))
            line.append(self.crit.index(fil.cp2))
            lines.append(line)
            start = end
        vp = tvtk.PolyData(points=points, verts=verts, lines=lines)

        # add point data arrays
        for i in range(self.ncrit_data):
            array = np.array([crit.data[i] for crit in self.crit])
            fillminone = np.empty(fil_npoints, dtype=array.dtype)
            fillminone.fill(-1)
            array = np.concatenate((array, fillminone))
            vp.point_data.add_array(array)
            vp.point_data.get_array(i).name = self.crit_data_name[i]
        array = np.array([crit.typ for crit in self.crit])
        fillminone = np.empty(fil_npoints, dtype=array.dtype)
        fillminone.fill(-1)
        array = np.concatenate((array, fillminone))
        vp.point_data.add_array(array)
        vp.point_data.get_array(i + 1).name = "critical_index"

        # add cell (lines) data array
        array = np.concatenate(
            (np.repeat(0, self.ncrit), np.arange(self.nfil)))
        vp.cell_data.add_array(array)
        vp.cell_data.get_array(0).name = "arc_id"

        array = np.concatenate(
            (np.repeat(1, self.ncrit), np.repeat(2, self.nfil)))
        vp.cell_data.add_array(array)
        vp.cell_data.get_array(1).name = "type"

        array = np.concatenate((np.repeat(-1, self.ncrit),
                                np.array([self.crit.index(fil.cp1) for fil in self.fil])))
        vp.cell_data.add_array(array)
        vp.cell_data.get_array(2).name = "down_index"

        array = np.concatenate((np.repeat(-1, self.ncrit),
                                np.array([self.crit.index(fil.cp2) for fil in self.fil])))
        vp.cell_data.add_array(array)
        vp.cell_data.get_array(3).name = "up_index"

        array = np.concatenate((np.repeat(-1, self.ncrit),
                                np.array([fil.nsamp for fil in self.fil])))
        vp.cell_data.add_array(array)
        vp.cell_data.get_array(4).name = "length"

        array = np.concatenate((np.array([c.boundary for c in self.crit]),
                                np.array([fil.cp1.boundary | fil.cp2.boundary for fil in self.fil])))
        vp.cell_data.add_array(array)
        vp.cell_data.get_array(5).name = "flags"

        print("Writing skeleton vtp file {0} \n".format(filename))

        v = tvtk.XMLPolyDataWriter()
        v.set_input_data(vp)
        v.file_name = filename
        v.write()

    def write_crits(self, filename):
        with open(filename, 'w') as f:
            print("Writing ascii .crits file {0} \n".format(filename))
            f.write("#critical points\n")
            f.write(
                "#X0 X1 X2 value type pair_id boundary persistence persistence_nsigma persistence_ratio\n")
            f.write("#3 {0:d}\n".format(self.ncrit))
            persistence_ratio_id = self.crit_data_name.index(
                'persistence_ratio')
            persistence_nsig_id = self.crit_data_name.index(
                'persistence_nsigmas')
            persistence_id = self.crit_data_name.index('persistence')
            for crit in self.crit:
                values = list(crit.pos)
                try:
                    pair_id = self.crit.index(crit.pair)
                except ValueError:
                    pair_id = -1
                values.extend([crit.val, crit.typ, pair_id, crit.boundary])
                values.append(crit.data[persistence_id])
                values.append(crit.data[persistence_nsig_id])
                values.append(crit.data[persistence_ratio_id])
                f.write(" ".join(map(str, values)) + "\n")
            f.close()

# 2D SKELETON

    @property
    def is_broken_down2D(self):
        """Check if the breakdown option was set when running mse
        """
        try:
            return self._is_broken_down
        except AttributeError:
            typ3 = [c.typ for c in self.crit if c.typ == 3]
            if len(typ3) == 0:
                self._is_broken_down = False
            else:
                self._is_broken_down = True
            return self._is_broken_down

    def filter_nodes_alone2D(self, filter_filaments=True):
        """remove the nodes with only one filament connected to it
            and eventually removes the associated filaments
        """
        #alones_idx, alones_cp =zip(*[(i, cp) for i, cp in enumerate(self.crit) if cp.nfil==1 and cp.typ==3])
        alones_idx = [i for i, cp in enumerate(
            self.crit) if cp.nfil == 1 and cp.typ == 2]

        # mark bad maxima to be removed
        mask = np.ones(self.ncrit, dtype=bool)
        mask[alones_idx] = False

        if filter_filaments:
            mask_fil = np.ones(self.nfil, dtype=bool)

        for i in alones_idx:
            cp = self.crit[i]
            # remove persistence pair reference
            cp.pair.pair = None

            # remove the connected filament
            if filter_filaments:
                # mark the filament to be removed
                mask_fil[self.fil.index(cp.filId[0])] = False
                # remove the connection in the saddle
                saddle = cp.destCritId[0]
                Id_in_saddle = saddle.destCritId.index(cp)
                saddle.unconnect_fil(Id_in_saddle)
                # if saddle is alone remove it
                if saddle.nfil == 0:
                    mask[self.crit.index(saddle)] = False

        # remove the maxima
        self.crit[:] = np.array(self.crit)[mask]

        # remove all marked filaments
        if filter_filaments:
            self.fil[:] = np.array(self.fil)[mask_fil]

    def filaments_from_node2D(self, crit_index):
        '''return the list of filaments and saddles connected to node
        This routine is useful in the case of broken skeleton (option -breakdown of skelconv)
        otherwise it is straightforward.
        Each filament is returned as a list of broken filament
        (ie with bifurcation point at extremities)
        '''
        p = self.crit[crit_index]
        if p.typ != 2:
            raise SkelError('wrong type, max point expected (type 3)')
        all_fil_list = []
        all_saddle_list = []
        for i in range(p.nfil):
            fil = p.filId[i]
            dest = p.destCritId[i]
            res, saddle = self.follow_filament_to_cp(dest, fil, node=False)
            if res != None:
                all_fil_list.append(res)
                all_saddle_list.append(saddle)  # several may be returned
        return all_fil_list, all_saddle_list

    def distance_to_node2D(self, crit_index, points):
        """compute the distance of a given point to
        a given node (critical point 3)"""
        # nodes = (x for x in self.crit if x.typ==3)
        p = self.crit[crit_index]
        if p.typ != 2:
            raise SkelError('wrong type, max point expected (type 2)')
        d = np.sqrt((p.pos[0] - points[:, 0])**2 +
                    (p.pos[1] - points[:, 1])**2)
        return d

    def distance_to_nearest_node2D(self, points):
        """compute the distance of a given point to
        nearest node (critical point 3)"""
        # nodes = (x for x in self.crit if x.typ==3)
        try:
            tree = self._node_tree
        except AttributeError:
            nodes_pos = np.array([x.pos for x in self.crit if x.typ == 2])
            self._node_tree = KDTree(nodes_pos)
            tree = self._node_tree
        d, idx = tree.query(points)
        crits_id = np.array([i for i, x in enumerate(self.crit) if x.typ == 2])
        return d, crits_id[idx]

    def distance_to_nearest_node3Dproj(self, points):
        """compute the distance of a given point to
        nearest node (critical point 3)"""
        # nodes = (x for x in self.crit if x.typ==3)
        try:
            tree = self._node_tree
        except AttributeError:
            nodes_pos = np.array([x.pos for x in self.crit if x.typ == 3])
            self._node_tree = KDTree(nodes_pos)
            tree = self._node_tree
        d, idx = tree.query(points)
        crits_id = np.array([i for i, x in enumerate(self.crit) if x.typ == 3])
        return d, crits_id[idx]

    def distance_to_nearest_saddle2D(self, points):
        """compute the distance of a given point to
        nearest saddle-2 (critical point 2)"""
        # nodes = (x for x in self.crit if x.typ==3)
        try:
            tree = self._saddle_tree
        except AttributeError:
            saddles_pos = np.array([x.pos for x in self.crit if x.typ == 1])
            tree = self._saddle_tree = KDTree(saddles_pos)
        d, idx = tree.query(points)
        saddles_id = np.array(
            [i for i, x in enumerate(self.crit) if x.typ == 1])
        return d, saddles_id[idx]

    def filaments_from_saddle2D(self, crit_index):
        p = self.crit[crit_index]
        if p.typ != 1:
            raise SkelError('wrong type, saddle point expected (type 1)')
        all_fil_list = []
        all_node_list = []
        for i in range(p.nfil):
            #print('tracing filament {0}/{1}'.format(i,p.nfil))
            fil = p.filId[i]
            dest = p.destCritId[i]
            res, node = self.follow_filament_to_cp(dest, fil)
            if res != None:
                all_fil_list.append(res)
                all_node_list.append(node)  # only one is returned
        return all_fil_list, all_node_list

    def filaments_from_node2D(self, crit_index):
        '''return the list of filaments and saddles connected to node
        This routine is useful in the case of broken skeleton (option -breakdown of skelconv)
        otherwise it is straightforward.
        Each filament is returned as a list of broken filament
        (ie with bifurcation point at extremities)
        '''
        p = self.crit[crit_index]
        if p.typ != 2:
            raise SkelError('wrong type, max point expected (type 2)')
        all_fil_list = []
        all_saddle_list = []
        for i in range(p.nfil):
            fil = p.filId[i]
            dest = p.destCritId[i]
            res, saddle = self.follow_filament_to_cp(dest, fil, node=False)
            if res != None:
                all_fil_list.append(res)
                all_saddle_list.append(saddle)  # several may be returned
        return all_fil_list, all_saddle_list

    def connectivity2D(self, crit_index, R):
        """computes connectivity as the number of filaments crossing 1.5 x R200"""
        p = self.crit[crit_index]
        if p.typ != 2:
            raise SkelError('wrong type, max point expected (type 3)')
        fil_lst = np.array(self.filaments_from_node2D(crit_index)[0]).flatten()
        c2d = 0
        for fil in fil_lst:
            ctmp = 0
            l = 0
            while ctmp < 1 and l < len(fil.points[:-1]):
                p1 = fil.points[l]
                p2 = fil.points[l + 1]
                d1 = np.sqrt((p.pos[0] - p1[0])**2 + (p.pos[1] - p1[1])**2)
                d2 = np.sqrt((p.pos[0] - p2[0])**2 + (p.pos[1] - p2[1])**2)
                if d1 < R and d2 > R:
                    c2d += 1
                    ctmp += 1
                if d2 < R and d1 > R:
                    c2d += 1
                    ctmp += 1
                l = l + 1
        return c2d

    def fof_arround_max2D(self, delaunay_cat, fieldname, densfrac=.1, fof_max=30):
        """compute fof,
                starting from max,
                stopping at the density fraction densfrac between max and the highest connected saddle.
                (densfrac=0 means stop at the density of the saddle)
        """
        print("Computing fof arround max")
        typ2Id = np.array([i for i, x in enumerate(self.crit) if x.typ == 2])
        tree = KDTree(delaunay_cat.points)
        pos2 = np.array([self.crit[i].pos for i in typ2Id])
        pos2 = np.c_[pos2, np.zeros(len(pos2))]
        _, nearest_id = tree.query(pos2)
        field_value = delaunay_cat.point_data.get_array(fieldname).to_array()
        fof_indices = np.empty(delaunay_cat.number_of_points, dtype=int)
        fof_indices.fill(-1)
        fof_size = np.empty(typ2Id.size, dtype=int)
        fof_size.fill(-1)
        for i, Id in enumerate(typ2Id):
            cells = delaunay_cat.get_cells()
            # print "\ncrit point {0}".format(i)
            cp = self.crit[Id]
            # get all the connected saddle and set the threshold to the highest
            _, saddle_lst = self.filaments_from_node2D(Id)
            if not saddle_lst:
                continue  # no filaments from this node...
            saddle_dens = np.array(
                [x.val for x in mpu.datastructures.flatten(saddle_lst)])
            density_thres = saddle_dens.max()
            density_thres += densfrac * (cp.val - density_thres)
            # follow delaunay graph for nearest points above thres
            #####################################################
            # start the point set with nearest neighbor of node
            point_set = np.array([nearest_id[i]])
            curpoint = 0
            while curpoint < point_set.size and point_set.size < fof_max:
                # print curpoint,'/',point_set.size
                # find cells containing current point
                cellsId = (np.where(cells == point_set[curpoint]))[0]
                # add in the set the points of the selected cells and above
                # density threshold
                if cellsId.size != 0:
                    points = np.unique(np.take(cells, cellsId, axis=0))
                    points_val = np.take(field_value, points)
                    # & (points_val < cp.val)
                    points = np.compress(points_val > density_thres, points)
                    point_new = np.setdiff1d(
                        points, point_set, assume_unique=True)
                    point_set = np.append(point_set, point_new)
                    np.delete(cells, cellsId, axis=0)
                curpoint += 1
            if point_set.size >= fof_max:
                print("crit point {0}".format(i), "fof too big, stopping...")
            else:
                fof_indices[point_set] = i
                fof_size[i] = point_set.size
            # print "fof size: ", point_set.size
        return fof_indices, fof_size
