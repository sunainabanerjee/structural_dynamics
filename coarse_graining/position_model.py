import os
import math
import logging
import numpy as np
from geometry import *
from regressor import *
from structural_dynamics import *
from .cg_properties import *
from .generate_signature import *


__version__ = "1.0"
__all__ = ['get_model_dihedral_signature',
           'PositionPredictor']


def get_model_dihedral_signature(pdb,
                                 resids,
                                 properties=CoarseGrainProperty.default_properties(),
                                 topn=12):
    assert isinstance(pdb, PDBStructure)
    assert isinstance(resids, list)
    all_residues = pdb.residue_ids
    resids = [r for r in resids if r in all_residues]
    assert len(resids) > 0
    proper_ids, dihed_evals = [], []
    for r in resids:
        amino = get_amino(pdb.residue_name(r))
        diheds = AminoBuilder.model_dihedrals(pdb.residue_name(r))
        atom_list = pdb.atom_names(r)
        full_list = amino.atom_names()
        complete = all([atm in atom_list  for atm in full_list])
        idx = all_residues.index(r)
        if complete and (idx > 1):
            proper_ids.append(r)
            evals = []
            for dihed in diheds:
                atm1, resid1 = dihed[0][0], all_residues[idx + dihed[0][1]]
                atm2, resid2 = dihed[1][0], all_residues[idx + dihed[1][1]]
                atm3, resid3 = dihed[2][0], all_residues[idx + dihed[2][1]]
                atm4, resid4 = dihed[3][0], all_residues[idx + dihed[3][1]]
                c1 = Coordinate3d(*pdb.xyz(resid1, atm1))
                c2 = Coordinate3d(*pdb.xyz(resid2, atm2))
                c3 = Coordinate3d(*pdb.xyz(resid3, atm3))
                c4 = Coordinate3d(*pdb.xyz(resid4, atm4))
                d = dihedral(c1, c2, c3, c4)
                if d < 0:
                    d = d + 2 * math.pi
                evals.append(d)
            dihed_evals.append(evals)
    if len(proper_ids) > 0:
        sig = cg_neighbor_signature(pdb,
                                    proper_ids,
                                    properties=properties,
                                    topn=topn)
        return proper_ids, sig, dihed_evals


class PositionPredictor:
    def __init__(self,
                 model_folder,
                 fmt_string="{}_pos.{}",
                 model_type='mlp'):
        assert os.path.isdir(model_folder)
        assert model_type in {'mlp', 'xgb'}
        ext = 'h5' if model_type == 'mlp' else 'dat'
        aminos = [get_amino(aa) for aa in valid_amino_acids()]
        self.__models = {}
        for aa in aminos:
            nn = aa.name(one_letter_code=False)
            model_file = os.path.join(model_folder,
                                      fmt_string.format(nn.lower(), ext))
            assert os.path.isfile(model_file)
            if model_type == 'mlp':
                self.__models[nn] = MLP()
            elif model_type == 'xgb':
                self.__models[nn] = XGBoost()
            self.__models[nn].load(model_file)

    def predict_position(self, ca_trace):
        recons = ProteinReconstruction(ca_trace)
        last_r, diheds, counter = None, [], 0
        while not recons.is_complete():
            r = recons.curr_fix_residue()
            if r != last_r:
                last_r = r
                aa = ca_trace.get_amino(r).name(one_letter_code=False)
                sig = cg_neighbor_signature(ca_trace, [r])[0]
                n = len(sig)
                sig = np.array(sig, dtype=np.float).reshape((1, n))
                diheds = self.__models[aa].predict(sig)[0]
                counter = 0
            print(diheds)
            recons.fix(diheds[counter])
            counter += 1
        return recons.get_pdb()
