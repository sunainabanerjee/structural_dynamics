import os
import logging
from geometry import *
from mutant_model import *
from structural_dynamics import *


if __name__ == "__main__":
    logging.basicConfig(debug=logging.DEBUG)
    ref_pdb_file = os.path.join(os.path.dirname(__file__), 'data', 'structure_1a1v.pdb')
    mutant_trj_file = os.path.join(os.path.dirname(__file__), 'data', 'ex_trajectory_1.pdb')
    wild_trj_file = os.path.join(os.path.dirname(__file__), 'data', 'ex_trajectory_2.pdb')

    assert os.path.isfile(ref_pdb_file)
    ref_trace = pdb_to_catrace(read_pdb(pdb_file=ref_pdb_file)[0]['A'])
    max_crd, min_crd = Coordinate3d(28.0, 54.0, 32.0), Coordinate3d(4.0, 11.0, 18.0)
    for i in range(len(max_crd)):
        max_crd[i] += 2
        min_crd[i] -= 2

    assert os.path.isfile(mutant_trj_file) and os.path.isfile(wild_trj_file)
    mut_trj = [pair['A'] for pair in read_trajectory_catrace(mutant_trj_file)]
    wild_trj = [pair['A'] for pair in read_trajectory_catrace(wild_trj_file)]

    mut_trj, wild_trj = align_nma_trajectories(mut_trj, wild_trj, return_array=False)
    sig_gen = SignatureGenerator(min_crd, max_crd)
    msig = trajectory_signature(ref_trace, mut_trj, sig_gen)
    wsig = trajectory_signature(ref_trace, wild_trj, sig_gen)

    for p in msig.keys():
        sscore = signature_diff(msig[p], wsig[p])
        print(len(sscore))
        print(sscore)


