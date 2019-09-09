import os
import numpy as np
from coarse_graining import *
from structural_dynamics import *

__version__ = "1.0"

if __name__ == "__main__":
    pdbfile = os.path.join(os.path.dirname(__file__), 'data', 'structure.pdb')
    assert os.path.isfile(pdbfile)
    pdblist = read_pdb(pdbfile)
    assert len(pdblist) == 1
    pdb = pdblist[0]['B']
    amino_type = "W"
    residue_ids = [r for r in pdb.residue_ids if r > 0][:-1]
    residue_ids = [r for r in residue_ids if pdb.residue_name(r, one_letter_code=True) == amino_type]
    valid_residues = filter_complete_residues(pdb, residue_ids)
    ca_trace = pdb_to_catrace(pdb)
    nbr_sig = np.array(cg_neighbor_signature(ca_trace, valid_residues))
    vol = Volume3D(size=(3, 3, 3), dim=(5, 5, 5))
    sig = []
    for r in valid_residues:
        sig.append(vol.get_signature(pdb, r, reset=True))
    sig = np.concatenate((nbr_sig, np.array(sig)), axis=1)

