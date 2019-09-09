import os
import numpy as np
from coarse_graining import CoarseGrainProperty as CG
from structural_dynamics import *
from coarse_graining import *

if __name__ == "__main__":
    trj_file = os.path.join(os.path.dirname(__file__), 'data', 'wild_good_traj.pdb')
    ca_traj = [pair['A'] for pair in read_trajectory_catrace(trj_file)]
    fix_traj = [fix_ca_trace(traj) for traj in ca_traj]
    assert len(ca_traj) > 0
    protein_len = len(ca_traj[0])
    n_residues = 5
    residue_ids = ca_traj[0].residue_ids
    residues = [residue_ids[p] for p in [np.random.randint(low=0, high=protein_len) for i in range(n_residues)]]
    residues = sorted(list(set(residues)))
    for ca_trace in ca_traj:
        sig = cg_neighbor_signature(ca_trace, residues, properties=[CG.volume(), CG.distance()], topn=4)
        print(sig)


