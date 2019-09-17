import os
import math
import pandas as pd
from geometry import *
from structural_dynamics import *

if __name__ == "__main__":
    pdb_file = os.path.join(os.path.dirname(__file__),
                           "data",
                           "structure_1a1v.pdb")
    assert os.path.isfile(pdb_file)
    pdb = read_pdb(pdb_file=pdb_file)[0]['A']
    ca_trace = pdb_to_catrace(pdb)
    pr = ProteinReconstruction(ca_trace)

    for dihed in pr.dihedral_build_list():
        c1 = Coordinate3d(*pdb.xyz(dihed[0][1], dihed[0][0]))
        c2 = Coordinate3d(*pdb.xyz(dihed[1][1], dihed[1][0]))
        c3 = Coordinate3d(*pdb.xyz(dihed[2][1], dihed[2][0]))
        c4 = Coordinate3d(*pdb.xyz(dihed[3][1], dihed[3][0]))
        d = dihedral(c1, c2, c3, c4)
        print(dihed)
        pr.fix(d)
    pdb = pr.get_pdb()
    outfile = os.path.join(os.path.dirname(__file__), 'out', 'fixed.pdb')
    with open(outfile, "w") as fp:
        pdb.write(fp)

