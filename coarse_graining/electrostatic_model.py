import numpy as np
from .position_model import *
from structural_dynamics import *

__version__ = "1.0"
__all__ = ['ElectrostaticModel']


class ElectrostaticModel:
    def __init__(self,
                 model_folder,
                 fmt_string="{}_pos.{}",
                 model_type='xgb',
                 cutoff_distance=1.52,
                 dielectric = 1.0):
        assert cutoff_distance > 1e-3
        assert dielectric > 0
        self.__pos_model = PositionPredictor(model_folder=model_folder,
                                             fmt_string=fmt_string,
                                             model_type=model_type)
        self.__coords = None
        self.__charge = None
        self.__cutoff = cutoff_distance
        self.__epsilon = dielectric

    def add_protein(self, pdb):
        if isinstance(pdb, CaTrace):
            pdb = self.__pos_model.predict_position(ca_trace=pdb)
            residue_list = pdb.residue_ids
        else:
            residue_list = pdb.residue_ids[2:-1]
        assert isinstance(pdb, PDBStructure)
        coords, charge = [], []
        for r in residue_list:
            for aa in pdb.atom_names(r):
                coords.append([*pdb.xyz(r, aa)])
                charge.append(pdb.charge(r, aa))
        if len(coords) > 0:
            if self.__coords is None:
                self.__coords = np.array(coords, dtype=np.float)
                self.__charge = np.array(charge, dtype=np.float)
            else:
                self.__coords = np.concatenate((self.__coords,
                                                np.array(coords, dtype=np.float)),
                                               axis=0)
                self.__charge = np.concatenate((self.__charge,
                                                np.array(charge, dtype=np.float)),
                                               axis=0)

    def reset(self):
        self.__coords = None
        self.__charge = None

    def __len__(self):
        return 0 if self.__coords is None else self.__coords.shape[0]

    @property
    def xyz(self):
        return self.__coords

    @property
    def charge(self):
        return self.__charge

    def potential(self, x, y, z):
        assert self.__coords is not None
        crd = np.array([x, y, z], dtype=np.float)
        const = 1./(4. * np.pi * self.__epsilon)
        r = np.clip(np.sqrt(np.sum(np.square(self.__coords - crd),
                                   axis=1)),
                    a_min=self.__cutoff,
                    a_max=None)
        return np.sum(1./r) * const
