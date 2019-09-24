import os
import time
from mutant_model import *
from structural_dynamics import read_trajectory_catrace, read_pdb, pdb_to_catrace

if __name__ == "__main__":
    pdb_file = os.path.join(os.path.dirname(__file__), 'data', 'structure_1a1v.pdb')
    trj_file2 = os.path.join(os.path.dirname(__file__), 'data', 'ex_trajectory_2.pdb')
    assert os.path.isfile(pdb_file) and os.path.isfile(trj_file2)
    ref_pdb = pdb_to_catrace(read_pdb(pdb_file)[0]['A'])
    trj = [pair['A'] for pair in read_trajectory_catrace(trj_file2)]
    start = time.time()
    aln_trj = align_trajectory(ref_pdb, trj[0], trj, ca_trace=True)
    end = time.time()
    print("Alignment time: %f" % (end - start))
    out_file = os.path.join(os.path.dirname(__file__), 'out', 'aligned_to_1a1v.pdb')
    with open(out_file, "w") as fh:
        for i, snapshot in enumerate(aln_trj):
            fh.write("MODEL %6d\n" % (i+1))
            snapshot.write(fh)
            fh.write("ENDMDL\n")


