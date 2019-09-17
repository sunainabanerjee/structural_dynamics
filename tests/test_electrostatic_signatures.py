import os
import time
from geometry import *
from structural_dynamics import *
from coarse_graining import *

__version__ = "1.0"


if __name__ == "__main__":
    #aminos = [get_amino(aa) for aa in valid_amino_acids()]
    aminos = [get_amino(aa) for aa in ["R"]]
    model_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   '..',
                                   'scripts',
                                   'pos_models')
    pdbfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'data',
                           'structure_1a1v.pdb')
    pdb = read_pdb(pdbfile)[0]['A']
    ca_trace = pdb_to_catrace(pdb)
    all_residues = pdb.residue_ids
    coordinates, vols, resids = None, [], []
    prev = time.time()
    all_result = {}
    lines, counter = [], 0

    for amino in aminos:
        print("Resolving: %s" % amino)
        residues = [i for i in pdb.find_residue(amino)
                    if (all_residues.index(i) > 0) and
                    (all_residues.index(i) < len(all_residues) - 1)]
        rpm = ResiduePositionModel(amino, model_directory)
        result = rpm.get_position(ca_trace, residues)
        for r in result:
            result[r]["CA"] = Coordinate3d(*ca_trace.xyz(r))
            for aa in result[r]:
                line = "ATOM  %5d  %-3s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f " % (counter + 1,
                                                                                     aa,
                                                                                     amino,
                                                                                     "A",
                                                                                     r,
                                                                                     result[r][aa].x,
                                                                                     result[r][aa].y,
                                                                                     result[r][aa].z,
                                                                                     1.0,
                                                                                     1.0)
                lines.append(line)
                counter += 1

    outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "out",
                           "reconstructed.pdb")
    with open(outfile, "w") as fp:
        for line in lines:
            fp.write("%s\n" % line)
        fp.write("END\n")
