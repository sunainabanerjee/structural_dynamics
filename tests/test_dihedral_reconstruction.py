import os
import numpy as np
from geometry import *
from coarse_graining import *
from structural_dynamics import *


if __name__ == "__main__":
    s, n = 0, 3000
    for i in range(n):
        x = np.random.random((4, 3))*10 - 5
        dihed = dihedral(x[0, :], x[1, :], x[2, :], x[3, :])
        dist = distance(x[2, :], x[3, :])
        ang = angle(x[1, :], x[2, :], x[3, :])
        y = reconstruct_coordinate(x[0, :], x[1, :], x[2, :], dist, ang, dihed)
        s += distance(y, x[3, :])
    print("Reconstruction error from (%d) instances is (%.5f)" % (n, s))
    pdbfile = os.path.join( os.path.dirname(os.path.abspath(__file__)), 'data', 'structure.pdb')
    assert os.path.isfile(pdbfile)
    pdb = read_pdb(pdbfile)[0]['B']
    sig = get_atom_position_signature(pdb, get_amino('K'))
    for aa in sig.keys():
        print(sig[aa])

