
from scipy.spatial import cKDTree as KDTree
from tvtk.api import tvtk
import numpy as np


class VoidsRegion():

    def __init__(self, filename):
        self._read(filename)

    def _read(self, filename):
        v = tvtk.XMLUnstructuredGridReader(file_name=filename)
        v.update()
        self.voids = v.output

    def get_cells(self):
        cells = self.voids.get_cells().to_array().astype(int)
        cells.shape = (cells.size // 5, 5)
        cells = cells[:, 1:5]  # remove type of cells column
        return cells

    def indices_at_points(self):
        index = self.voids.point_data.get_array('index')
        if index:
            index = index.to_array().astype(int)
        else:
            index = np.arange(self.voids.number_of_points, dtype=int)
        true_index = self.voids.point_data.get_array('true_index')
        if true_index:
            true_index = true_index.to_array().astype(int)
            if not all(index == true_index):
                raise Exception("Error still guards in the voids")
        source_index = self.voids.cell_data.get_array('source_index')
        source_index = source_index.to_array().astype(int)
        voids_index = np.zeros(np.max(index) + 1, dtype=int) - 1
        # cell data to point data
        for p in range(self.voids.number_of_points):
            il = tvtk.IdList()
            self.voids.get_point_cells(p, il)
            if all(source_index[il] == np.array(source_index[il[0]])):
                voids_index[index[p]] = source_index[il[0]]
        return voids_index

    def volumes_and_mean_densities(self):
        source_index = self.voids.cell_data.get_array('source_index')
        source_index = source_index.to_array().astype(int)
        usi = np.unique(source_index)
        nb_voids = usi.size
        voids_dens = np.zeros(nb_voids)
        voids_vol = np.zeros(nb_voids)
        for i, si in enumerate(usi):
            p = tvtk.IdList()
            cellsid, = np.where(source_index == np.array(si))
            for icell in cellsid:
                cell = self.voids.get_cell(icell)
                vol = cell.compute_volume(*cell.points)
                voids_vol[i] += vol
                p.extend(cell.point_ids)
            p = np.unique(p)
            voids_dens[i] = p.size / voids_vol[i]
        return voids_vol, voids_dens, usi


class NodesRegion():

    def __init__(self, filename, vr):
        self._read(filename)
        if isinstance(vr, VoidsRegion):
            self.vr = vr
        else:
            raise SkelError(
                'A VoidRegion is needed to initialize a NodeRegion')

    def _read(self, filename):
        v = tvtk.XMLUnstructuredGridReader(file_name=filename)
        v.update()
        self.nodes = v.output

    def indices_at_points(self):
        index = self.nodes.point_data.get_array('index')
        if index:
            index = index.to_array().astype(int)
        else:
            index = np.arange(self.nodes.number_of_points, dtype=int)
        true_index = self.nodes.point_data.get_array('true_index')
        if true_index:
            true_index = true_index.to_array().astype(int)
            if not all(index == true_index):
                raise Exception("Error still guards in the nodes")
        source_index = self.nodes.point_data.get_array('source_index')
        source_index = source_index.to_array().astype(int)
        nodes_index = np.zeros(np.max(index) + 1, dtype=int) - 1
        nodes_index[index] = source_index
        return nodes_index

    def volumes_and_mean_densities(self):
        cells = self.vr.get_cells()
        index_vr = self.vr.voids.point_data.get_array('index')
        if index_vr:
            index_vr = index_vr.to_array().astype(int)
        else:
            index_vr = np.arange(self.vr.voids.number_of_points, dtype=int)
        cells = index_vr[cells]
        source_index = self.nodes.point_data.get_array('source_index')
        source_index = source_index.to_array().astype(int)
        index = self.nodes.point_data.get_array('index')
        if index:
            index = index.to_array().astype(int)
        else:
            index = np.arange(self.nodes.number_of_points, dtype=int)
        usi = np.unique(source_index)
        nb_nodes = usi.size
        nodes_dens = np.zeros(nb_nodes)
        nodes_vol = np.zeros(nb_nodes)
        for i, si in enumerate(usi):
            points, = np.where(source_index == np.array(si))
            points = index[points]
            cellsid = np.in1d(cells[:, 0], points)
            cellsid &= np.in1d(cells[:, 1], points)
            cellsid &= np.in1d(cells[:, 2], points)
            cellsid &= np.in1d(cells[:, 3], points)
            for icell in np.flatnonzero(cellsid):
                cell = self.vr.voids.get_cell(icell)
                vol = cell.compute_volume(*cell.points)
                nodes_vol[i] += vol
            if nodes_vol[i]:
                nodes_dens[i] = points.size / nodes_vol[i]
        return nodes_vol, nodes_dens, usi


class Walls():

    def __init__(self, filename):
        v = tvtk.XMLUnstructuredGridReader(file_name=filename)
        v.update()
        self.walls = v.output

    def _compute_centers(self):
        self._centers = np.zeros((self.walls.number_of_cells, 3))
        for i in range(self.walls.number_of_cells):
            cell = self.walls.get_cell(i)
            cell.triangle_center(cell.points[0], cell.points[1], cell.points[2],
                                 self._centers[i])

    @property
    def centers(self):
        try:
            return self._centers
        except AttributeError:
            self._compute_centers()
            return self._centers

    def distance(self, points):
        try:
            distances, indexes = self._tree.query(points)
        except AttributeError:
            self._tree = KDTree(self.centers)
            distances, indexes = self._tree.query(points)

        source_index = self.walls.cell_data.get_array('source_index')
        source_index = source_index.to_array().astype(int)
        wall_ids = source_index.take(indexes)

        return distances, wall_ids, indexes

    def _compute_surfaces(self):
        source_index = self.walls.cell_data.get_array('source_index')
        source_index = source_index.to_array().astype(int)
        usi = np.unique(source_index)
        self._surfaces = np.zeros(usi.size)
        for i, si in enumerate(usi):
            cellsid, = np.where(source_index == np.array(si))
            for icell in cellsid:
                cell = self.walls.get_cell(icell)
                self._surfaces[i] += cell.compute_area()

    @property
    def surfaces(self):
        try:
            return self._surfaces
        except AttributeError:
            self._compute_surfaces()
            return self._surfaces

    @property
    def total_surface(self):
        return np.sum(self.surfaces)
