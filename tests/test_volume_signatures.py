import os
import time
import numpy as np
from structural_dynamics import *
from coarse_graining import *

__version__ = "1.0"


if __name__ == "__main__":
    aminos = [get_amino(aa) for aa in valid_amino_acids()]
    model_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   '..',
                                   'scripts',
                                   'volume_models')
    rvm = VolumePredictor(model_directory, model_type='xgb')
    pdbfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'data',
                           'structure_1a1v.pdb')
    pdb = read_pdb(pdbfile)[0]['A']
    all_residues = pdb.residue_ids[1:-1]
    coordinates, vols, resids = None, [], []
    prev = time.time()
    ca_trace = pdb_to_catrace(pdb)
    recons = rvm.predict_volume(ca_trace=ca_trace, residue_ids=all_residues)
    for i, pos in enumerate(recons):
        if coordinates is None:
            coordinates = pos[1]
        else:
            coordinates = np.concatenate((coordinates, pos[1]), axis=0)
        vols = vols + pos[2]
        for j in range(pos[1].shape[0]):
            resids.append((pos[0],
                           ca_trace.get_amino(pos[0]).name(one_letter_code=False)))
    print(time.time() - prev)
    lines = []
    for i in range(coordinates.shape[0]):
        line = "ATOM  %5d  %-3s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f " % (i + 1,
                                                                             "SPH",
                                                                             "GRD",
                                                                             "A",
                                                                             resids[i][0],
                                                                             coordinates[i, 0],
                                                                             coordinates[i, 1],
                                                                             coordinates[i, 2],
                                                                             1.0,
                                                                             vols[i])
        lines.append(line)
    outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "out",
                           "pseudo_xgb.pdb")
    print("Writing [%s]" % outfile)
    with open(outfile, "w") as fp:
        for line in lines:
            fp.write("%s\n" % line)
        fp.write("END\n")
