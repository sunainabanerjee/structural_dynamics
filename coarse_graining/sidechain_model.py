import numpy as np
from geometry import *
from copy import deepcopy
from structural_dynamics import *
from .generate_signature import *

__version__ = "1.0"
__all__ = ['check_missing_atoms',
           'ordered_atom_xyz',
           'aligning_matrix',
           'placement_matrix',
           'find_axis_vectors',
           'place_atoms',
           'aligned_coordinates',
           'list_aligned_coordinates',
           'filter_complete_residues',
           'BatchSideChainConfigHandler']


def check_missing_atoms(pdb, resid):
    assert isinstance(pdb, PDBStructure)
    if resid not in pdb.residue_ids:
        return True
    atom_required = AminoAcid(pdb.residue_name(resid, one_letter_code=True)).atom_names()
    atom_present = pdb.atom_names(resid)
    return not all([aa in atom_present for aa in atom_required])


def ordered_atom_xyz(pdb, resid):
    if check_missing_atoms(pdb, resid):
        return []
    amino = AminoAcid(pdb.residue_name(resid, one_letter_code=True))
    return [list(pdb.xyz(resid, atom)) for atom in amino.atom_names()]


def find_axis_vectors(ca_trace, resid):
    assert isinstance(ca_trace, CaTrace)
    all_residues = ca_trace.residue_ids
    assert resid in all_residues
    i = all_residues.index(resid)
    assert (i > 0) and (i < len(all_residues) - 1)
    ca_prev = Coordinate3d(*ca_trace.xyz(all_residues[i - 1]))
    ca_curr = Coordinate3d(*ca_trace.xyz(all_residues[i]))
    ca_next = Coordinate3d(*ca_trace.xyz(all_residues[i + 1]))
    uv = connecting_vector(ca_prev, ca_curr).unit_vector
    wv = connecting_vector(ca_next, ca_curr).unit_vector
    x_axis = (uv + wv).unit_vector
    y_axis = crossp(uv, wv).unit_vector
    z_axis = crossp(x_axis, y_axis).unit_vector
    return x_axis, y_axis, z_axis


def placement_matrix(pdb, resid):
    if isinstance(pdb, PDBStructure):
        ca_trace = pdb_to_catrace(pdb)
    else:
        ca_trace = deepcopy(pdb)
    assert isinstance(ca_trace, CaTrace)
    x_axis, y_axis, z_axis = find_axis_vectors(ca_trace, resid)
    return np.array([x_axis.tolist(),
                     y_axis.tolist(),
                     z_axis.tolist()]).transpose(), point_vector(Coordinate3d(*ca_trace.xyz(resid)))


def aligning_matrix(pdb, resid):
    if isinstance(pdb, PDBStructure):
        ca_trace = pdb_to_catrace(pdb)
    else:
        ca_trace = deepcopy(pdb)
    assert isinstance(ca_trace, CaTrace)
    x_axis, y_axis, z_axis = find_axis_vectors(ca_trace, resid)
    axes = np.array([x_axis.tolist(), y_axis.tolist(), z_axis.tolist()]).transpose()
    return np.linalg.inv(axes)


def place_atoms(ca_trace, model, amino):
    assert isinstance(ca_trace, CaTrace)
    assert isinstance(amino, AminoAcid)
    all_residues = ca_trace.residue_ids

    """truncating first and last residues"""
    amino_residues = [r for r in all_residues[1:-1] if ca_trace.get_amino(r) == amino]
    coordinates = dict()
    if len(amino_residues) > 0:
        atoms = amino.atom_names()
        signature = np.array(cg_neighbor_signature(ca_trace, amino_residues))
        result = model.predict(signature)
        n, w = result.shape
        for i in range(n):
            j = 0
            res_coords = []
            while j < w:
                r, theta, phi = result[i, j], result[i, j + 1], result[i, j + 2]
                x, y, z = spherical_to_cartesian(r, theta, phi)
                res_coords.append((x, y, z))
                j += 3
            res_coords = np.array(res_coords)
            rotmat, trans = placement_matrix(ca_trace, amino_residues[i])
            res_coords = np.matmul(rotmat, res_coords.transpose()).transpose() + trans.toarray()
            counter = 0
            atom_coords = {}
            for aa in atoms:
                if aa != "CA":
                    atom_coords[aa] = {'x': res_coords[counter, 0],
                                       'y': res_coords[counter, 1],
                                       'z': res_coords[counter, 2]}
                    counter += 1
            coordinates[amino_residues[i]] = atom_coords
    return coordinates


def aligned_coordinates(pdb, resid):
    assert isinstance(pdb, PDBStructure)
    xyz = np.array(ordered_atom_xyz(pdb, resid))
    if xyz.shape[0] == 0:
        return None
    ca_trace = pdb_to_catrace(pdb)
    rotmat = aligning_matrix(pdb, resid)
    x, y, z = ca_trace.xyz(resid)
    xyz = xyz - [x, y, z]
    return np.matmul(rotmat, xyz.transpose()).transpose()


def list_aligned_coordinates(pdb, residue_ids):
    assert isinstance(pdb, PDBStructure)
    conformation, valid_residues = [], []
    for r in residue_ids:
        result = aligned_coordinates(pdb, r)
        if result is not None:
            sig = []
            for i, row in enumerate(result):
                sig = sig + row.tolist()
            conformation.append(sig)
            valid_residues.append(r)
    return valid_residues, np.array(conformation)


def filter_complete_residues(pdb, residue_ids):
    assert isinstance(pdb, PDBStructure)
    return [r for r in residue_ids if not check_missing_atoms(pdb, r)]


class BatchSideChainConfigHandler:
    def __init__(self, ca_trace):
        assert isinstance(ca_trace, CaTrace)
        self.__ca_trace = ca_trace
        self.__residues = ca_trace.residue_ids
        self.__xyz = np.array([[*self.__ca_trace.xyz(r)] for r in self.__residues])
        u = self.__unit_vector(self.__xyz[1:-1, :] - self.__xyz[:-2, :])
        v = self.__unit_vector(self.__xyz[1:-1, :] - self.__xyz[2:, :])
        self.__x_axis = self.__unit_vector(u + v)
        self.__y_axis = self.__unit_vector(np.cross(u, v))
        self.__z_axis = self.__unit_vector(np.cross(self.__x_axis, self.__y_axis))

    @staticmethod
    def __unit_vector(u):
        assert len(u.shape) == 2
        return (u.transpose()/np.sqrt(np.sum(np.square(u), axis=1))).transpose()

    def placement_matrix(self, res_id):
        n = self.__residues.index(res_id)
        assert (n > 0) and (n < len(self.__residues)-1)
        return np.array([self.__x_axis[n-1, :],
                         self.__y_axis[n-1, :],
                         self.__z_axis[n-1, :]]).transpose(), point_vector(Coordinate3d(*self.__xyz[n]))

