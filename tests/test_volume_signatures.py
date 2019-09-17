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
    rvm = {}
    for amino in aminos:
        print("Loading: %s" % amino)
        rvm[amino.name()] = ResidueVolumeModels(amino, model_directory)
    pdbfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'data',
                           'structure_1a1v.pdb')
    pdb = read_pdb(pdbfile)[0]['A']
    all_residues = pdb.residue_ids
    coordinates, vols, resids = None, [], []
    prev = time.time()
    for amino in aminos:
        residues = [i for i in pdb.find_residue(amino)
                    if (all_residues.index(i) > 0) and
                    (all_residues.index(i) < len(all_residues) - 1)]
        recons = rvm[amino.name()].get_volume(pdb_to_catrace(pdb), residues)
        if len(recons) == 0:
            continue
        for i, pos in enumerate(recons):
            if coordinates is None:
                coordinates = pos[0]
            else:
                coordinates = np.concatenate((coordinates, pos[0]), axis=0)
            vols = vols + pos[1]
            for j in range(pos[0].shape[0]):
                resids.append((residues[i], amino.name(one_letter_code=False)))
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
                           "pseudo.pdb")
    with open(outfile, "w") as fp:
        for line in lines:
            fp.write("%s\n" % line)
        fp.write("END\n")
