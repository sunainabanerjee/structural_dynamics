import os
import logging
import numpy as np
from geometry import *
from regressor import *
from .generate_signature import *
from .sidechain_model import *
from structural_dynamics import *

__version__ = "1.0"
__all__ = ['Volume3D',
           'VolumeSignatureGrid',
           'VolumePredictor']


class VolumeSignatureGrid:
    @staticmethod
    def size():
        return 3.8, 3.8, 3.8

    @staticmethod
    def dim():
        return 5, 5, 5

    @staticmethod
    def cell_volume():
        return 3.8**3

    @staticmethod
    def length():
        return 5**3


class Volume3D:
    def __init__(self,
                 size,
                 dim,
                 center=(0, 0, 0)):
        assert isinstance(size, np.int) or len(size) == 3
        assert isinstance(dim, np.int) or len(dim) == 3
        assert isinstance(center, Coordinate3d) or len(center) == 3
        if isinstance(size, np.float):
            self.__sx, self.__sy, self.__sz = size, size, size
        else:
            self.__sx, self.__sy, self.__sz = size[0], size[1], size[2]

        if isinstance(dim, np.int):
            self.__nx, self.__ny, self.__nz = dim, dim, dim
        else:
            self.__nx, self.__ny, self.__nz = dim[0], dim[1], dim[2]

        self.__cx, self.__cy, self.__cz = center[0], center[1], center[2]
        self.__vfrac = np.zeros(self.length, dtype=np.float)
        self.__logger = logging.getLogger('coarse_graining.Volume3D')

    @property
    def size(self):
        return self.__sx, self.__sy, self.__sz

    @property
    def dim(self):
        return self.__nx, self.__ny, self.__nz

    @property
    def length(self):
        return self.__nx * self.__ny * self.__nz

    def __len__(self):
        return self.length

    @property
    def cell_volume(self):
        return self.__sx * self.__sy * self.__sz

    @property
    def volume(self):
        return self.cell_volume * self.length

    @property
    def center(self):
        return Coordinate3d(self.__cx, self.__cy, self.__cz)

    def set_center(self, x, y, z):
        self.__cx, self.__cy, self.__cz = x, y, z

    def to_grid_coordinate(self, n):
        assert (n >= 0) and (n < self.length)
        gx, gy, gz = n // (self.__ny*self.__nz), (n // self.__nz) % self.__ny, n % self.__nz
        return gx, gy, gz

    def to_linear_index(self, gx, gy, gz):
        assert (gx >= 0) and (gx <= self.__nx)
        assert (gy >= 0) and (gy <= self.__ny)
        assert (gz >= 0) and (gz <= self.__nz)
        return gx * (self.__ny * self.__nz) + gy * self.__nz + gz

    @property
    def x_span(self):
        return (self.__nx * self.__sx)/2.0

    @property
    def y_span(self):
        return (self.__ny * self.__sy) / 2.0

    @property
    def z_span(self):
        return (self.__nz * self.__sz) / 2.0

    @property
    def x_max(self):
        return self.__cx + self.x_span

    @property
    def x_min(self):
        return self.__cx - self.x_span

    @property
    def y_max(self):
        return self.__cy + self.y_span

    @property
    def y_min(self):
        return self.__cy - self.y_span

    @property
    def z_max(self):
        return self.__cz + self.z_span

    @property
    def z_min(self):
        return self.__cz - self.z_span

    @property
    def x_range(self):
        return self.x_min, self.x_max

    @property
    def y_range(self):
        return self.y_min, self.y_max

    @property
    def z_range(self):
        return self.z_min, self.z_max

    def cell_containing(self, crd):
        cell = []
        if self.is_inside(crd):
            gx, gy, gz = self.get_grid_coordinate(crd)
            cell.append([self.x_min + gx * self.__sx, self.x_min + (gx + 1)*self.__sx])
            cell.append([self.y_min + gy * self.__sy, self.y_min + (gy + 1)*self.__sy])
            cell.append([self.z_min + gz * self.__sz, self.z_min + (gz + 1)*self.__sz])
        return cell

    def is_inside(self, crd):
        assert isinstance(crd, Coordinate3d) or (len(crd) == 3)
        if (crd[0] > self.x_min) and (crd[0] <= self.x_max):
            if (crd[1] > self.y_min) and (crd[1] <= self.y_max):
                return (crd[2] > self.z_min) and (crd[2] <= self.z_max)
        return False

    def get_grid_coordinate(self, crd):
        assert isinstance(crd, Coordinate3d) or len(crd) == 3
        gx, gy, gz = None, None, None
        if self.is_inside(crd):
            gx = int((crd[0] - self.x_min) // self.__sx)
            gy = int((crd[1] - self.y_min) // self.__sy)
            gz = int((crd[2] - self.z_min) // self.__sz)
        return gx, gy, gz

    def cell_center(self, gx, gy=None, gz=None):
        if gy is None:
            assert (gx < self.length) and (gx >= 0)
            gx, gy, gz = self.to_grid_coordinate(gx)
        assert (gx >= 0) and (gx < self.__nx)
        assert (gy >= 0) and (gy < self.__ny)
        assert (gz >= 0) and (gz < self.__nz)
        x = self.x_min + (gx + 0.5)*self.__sx
        y = self.y_min + (gy + 0.5)*self.__sy
        z = self.z_min + (gz + 0.5)*self.__sz
        return x, y, z

    def reset(self):
        self.__vfrac = np.zeros(self.length, dtype=np.float)

    def add_volume(self, crd, volume):
        assert isinstance(crd, Coordinate3d) or len(crd) == 3
        if self.is_inside(crd) and (volume > 0):
            gx, gy, gz = self.get_grid_coordinate(crd)
            r_approx = equivalent_sphere_radius(volume)
            cell = self.cell_containing(crd)
            assert len(cell) == 3
            nbrs = [(gx, gy, gz)]

            """Checking whether the volume has intersection with neighboring cells"""
            if (crd[0] - cell[0][0]) < r_approx:
                if gx > 0:
                    nbrs.append((gx-1, gy, gz))
            if (cell[0][1] - crd[0]) < r_approx:
                if gx < self.__nx - 1:
                    nbrs.append((gx+1, gy, gz))

            if (crd[1] - cell[1][0]) < r_approx:
                if gy > 0:
                    nbrs.append((gx, gy-1, gz))
            if (cell[1][1] - crd[1]) < r_approx:
                if gy < self.__ny - 1:
                    nbrs.append((gx, gy+1, gz))

            if (crd[2] - cell[2][0]) < r_approx:
                if gz > 0:
                    nbrs.append((gx, gy, gz-1))
            if (cell[2][1] - crd[2]) < r_approx:
                if gz < self.__nz - 1:
                    nbrs.append((gx, gy, gz+1))

            """Adding contribution to neighboring cells"""
            r = equivalent_sphere_radius(self.cell_volume)
            for x, y, z in nbrs:
                i = self.to_linear_index(x, y, z)
                f = self.__vfrac[i]
                v = sphere_intersection_volume(r_approx, Coordinate3d(crd[0], crd[1], crd[2]),
                                               r, Coordinate3d(*self.cell_center(x, y, z)))
                inc_f = v / self.cell_volume
                if v > self.cell_volume:
                    self.__logger.warning("Element will not fit in one cell. "
                                          "Total volume accounted will be less than actual!!")
                if f + inc_f > 1.0:
                    self.__logger.warning("Volume exceeds cell volume, undercounting!!")
                self.__vfrac[i] = min(f + inc_f, 1.0)
            return True
        if volume <= 0:
            self.__logger.warning("Improper volume (%f)" % volume)
        else:
            self.__logger.warning("Coordinate outside box boundary!")
        return False

    def __cell_radius(self):
        return np.mean([self.__sx, self.__sy, self.__sz]) / 2.0

    def get_signature(self, pdb, residue_id, reset=False):
        if reset:
            self.reset()
        if isinstance(pdb, PDBStructure) and (residue_id in pdb.residue_ids):
            xyz = aligned_coordinates(pdb, residue_id)
            atoms = get_amino(pdb.residue_name(residue_id)).atom_names()
            if len(atoms) != xyz.shape[0]:
                raise Exception("Missing atoms in the residue")
            assert xyz.shape[1] == 3
            if reset is True:
                self.reset()
            for i in range(xyz.shape[0]):
                if self.is_inside(xyz[i, :]):
                    r = get_vdw_radius(atoms[i])
                    self.add_volume(xyz[i, :], sphere_volume(r))
        return self.__vfrac


class VolumePredictor:
    def __init__(self,
                 model_folder,
                 fmt_string = "{}_volume.{}",
                 model_type='mlp',
                 min_occupancy=0.05):
        assert os.path.isdir(model_folder)
        assert model_type in {'xgb', 'mlp'}
        ext = 'h5' if model_type == 'mlp' else 'dat'
        aminos = [get_amino(aa) for aa in valid_amino_acids(one_letter=False)]
        self.__models = {}
        self.__min_occupancy = min_occupancy
        for aa in aminos:
            name = aa.name(one_letter_code=False)
            model_file = os.path.join(model_folder,
                                      fmt_string.format(name.lower(), ext))
            assert os.path.isfile(model_file)
            if model_type == 'xgb':
                self.__models[name] = XGBoost()
            elif model_type == 'mlp':
                self.__models[name] = MLP()
            self.__models[name].load(model_file)

    def predict_volume(self, ca_trace, residue_ids):
        assert isinstance(ca_trace, CaTrace)
        assert isinstance(residue_ids, list)
        all_residues = ca_trace.residue_ids
        residue_ids = [r for r in residue_ids if r in all_residues]
        signatures = np.array(cg_neighbor_signature(ca_trace, residues=residue_ids))
        grouped_signatures, grouped_resids = {}, {}
        for i, r in enumerate(residue_ids):
            aa = ca_trace.get_amino(r).name(one_letter_code=False)
            if aa not in grouped_signatures:
                grouped_signatures[aa] = []
                grouped_resids[aa] = []
            grouped_signatures[aa].append(signatures[i])
            grouped_resids[aa].append(r)
        grid = Volume3D(size=VolumeSignatureGrid.size(),
                        dim=VolumeSignatureGrid.dim())
        reconstruction = []
        for aa in grouped_signatures.keys():
            preds = self.__models[aa].predict(np.array(grouped_signatures[aa],
                                                       dtype=np.float))
            res_lst = grouped_resids[aa]
            for i, r in enumerate(res_lst):
                m, v = placement_matrix(ca_trace, r)
                coordinates, vol = list(), list()
                for j in range(preds.shape[1]):
                    if preds[i, j] >= self.__min_occupancy:
                        coordinates.append(grid.cell_center(j))
                        vol.append(preds[i, j] * VolumeSignatureGrid.cell_volume())
                coordinates = np.matmul(m, np.array(coordinates).transpose()).transpose()
                coordinates = coordinates + v.toarray()
                reconstruction.append((r, coordinates, vol))
        reconstruction.sort(key=lambda x: x[0])
        return reconstruction




