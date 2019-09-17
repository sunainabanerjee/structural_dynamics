import os
import numpy as np
from geometry import *
from regressor import *
from .generate_signature import *
from structural_dynamics import *
from .cg_properties import CoarseGrainProperty as CG

__version__ = "1.0"
__all__ = ['pseudo_atom',
           'get_configurations',
           'get_atom_position_signature',
           'ResiduePositionModel']


def pseudo_atom(c1, c2, c3, d=1.0):
    assert isinstance(c1, Coordinate3d) or len(c1) == 3
    assert isinstance(c2, Coordinate3d) or len(c2) == 3
    assert isinstance(c3, Coordinate3d) or len(c3) == 3
    w = connecting_vector(c1, c2).unit_vector + connecting_vector(c3, c2).unit_vector
    return Coordinate3d(c3.x + w.x * d,
                        c3.y + w.y * d,
                        c3.z + w.z * d)


def get_configurations(pdb, resid):
    assert isinstance(pdb, PDBStructure)
    all_residues = pdb.residue_ids
    assert resid in all_residues
    idx = all_residues.index(resid)
    assert (idx > 0) and (idx < len(all_residues))
    atoms_prev = pdb.atom_names(all_residues[idx-1])
    atoms_curr = pdb.atom_names(all_residues[idx])
    atoms_next = pdb.atom_names(all_residues[idx+1])
    config = []
    if ('CA' in atoms_prev) and ('CA' in atoms_curr) and ('CA' in atoms_next):
        c1 = Coordinate3d(*pdb.xyz(all_residues[idx-1], "CA"))
        c2 = Coordinate3d(*pdb.xyz(all_residues[idx], "CA"))
        c3 = Coordinate3d(*pdb.xyz(all_residues[idx+1], "CA"))
        c4 = pseudo_atom(c1, c2, c3)
        atom_order = get_amino(pdb.residue_name(residue_id=resid)).atom_names()
        for aname in atom_order:
            if (aname != "CA") and (aname in atoms_curr):
               cn = Coordinate3d(*pdb.xyz(resid, aname))
               dist = distance(c4, cn)
               ang = angle(c3, c4, cn)
               dihed = dihedral(c2, c3, c4, cn)
               config.append((aname, dist, ang, dihed))
    return config


def get_atom_position_signature(pdb,
                                amino,
                                properties=CG.default_properties(),
                                topn=12):
    assert isinstance(pdb, PDBStructure) and (amino, AminoAcid)
    assert all([p in CG.supported_properties() for p in properties]) and topn > 0
    all_residues = pdb.residue_ids
    residues = pdb.find_residue(aa_type=amino)
    signature = dict()
    selected_residues = []
    for r in residues:
        i = all_residues.index(r)
        if (i > 0) and (i < len(all_residues) - 1):
            config = get_configurations(pdb, r)
            signature[r] = {}
            for aa, dist, ang, dihed in config:
                assert dist >= 0
                ang = 2*np.pi + ang if ang < 0 else ang
                dihed = 2*np.pi + dihed if dihed < 0 else dihed
                signature[r][aa] = [dist, ang, dihed]
            selected_residues.append(r)
    sig = cg_neighbor_signature(pdb,
                                selected_residues,
                                properties=properties,
                                topn=topn)
    atom_signature = {}
    if len(sig) == len(signature):
        for i, r in enumerate(selected_residues):
            for aa in signature[r].keys():
                if aa not in atom_signature:
                    atom_signature[aa] = []
                atom_signature[aa].append(sig[i] + signature[r][aa])
    return atom_signature


class ResiduePositionModel:
    def __init__(self, amino, model_directory, model_type='mlp'):
        assert isinstance(amino, AminoAcid)
        assert os.path.isdir(model_directory)
        assert model_type in {'mlp', 'xgb'}
        amino_name = amino.name(one_letter_code=False).lower()
        atom_names = [aa for aa in amino.atom_names() if aa != "CA"]
        ext = 'h5' if model_type == 'mlp' else 'dat'
        self.__models = {}
        for aa in atom_names:
            model_file = os.path.join(model_directory,
                                      '%s_%s_pos.%s' % (amino_name, aa, ext))
            assert os.path.isfile(model_file)
            if model_type == 'mlp':
                self.__models[aa] = MLP()
            else:
                self.__models[aa] = XGBoost()
            self.__models[aa].load(model_file)
        self.__amino = amino

    @property
    def atoms(self):
        return list(self.__models.keys())

    def __getitem__(self, key):
        if key not in self.__models:
            raise IndexError("Missing key (%s)" % key)
        return self.__models[key]

    def fix(self, ca_trace, r):
        assert isinstance(ca_trace, CaTrace)
        all_residues = ca_trace.residue_ids
        i = all_residues.index(r)
        assert (i > 0) and (i < len(all_residues))
        c1 = Coordinate3d(*ca_trace.xyz(all_residues[i-1]))
        c2 = Coordinate3d(*ca_trace.xyz(all_residues[i]))
        c3 = Coordinate3d(*ca_trace.xyz(all_residues[i+1]))
        p = pseudo_atom(c1, c2, c3)
        nbr = np.array(cg_neighbor_signature(ca_trace, [r]))
        residue = {}
        for aa in self.__models:
            pred = self.__models[aa].predict(nbr)[0]
            residue[aa] = reconstruct_coordinate(c1, c2, p, pred[0], pred[1], pred[2])
        return residue

    def get_position(self, ca_trace, residue_ids):
        assert isinstance(ca_trace, CaTrace)
        assert isinstance(residue_ids, list)
        result = {}
        for r in residue_ids:
            assert ca_trace.get_amino(r) == self.__amino
            result[r] = self.fix(ca_trace, r)
        return result

