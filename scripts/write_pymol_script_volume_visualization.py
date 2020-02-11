import os
import argparse
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Produce pymol script for a volume visualization")

    parser.add_argument('--pdb-file', action='store', dest='pdb_file',
                        type=str, required=True,
                        help="target pdb file against which the prediction will be performed")

    parser.add_argument('--volume-desc', action='store', dest='vol_desc',
                        type=str, required=True,
                        help="volume description file as produced by the coarse grained model")

    parser.add_argument("--chain", action='store', dest='chain',
                        type=str, required=False, default='A',
                        help="PDB chain id")

    parser.add_argument("--residue", action='store', dest='residue',
                        type=int, required=True,
                        help="Residue Id of the target protein for the given description file")

    parser.add_argument("--out", action='store', dest='out_pml',
                        type=str, required=True,
                        help="Out file location")

    args = parser.parse_args()

    if not os.path.isfile(args.pdb_file):
        raise Exception("Can not access PDB file [%s]" % args.pdb_file)

    if not os.path.isfile(args.vol_desc):
        raise Exception("Can not access volume description file [%s]" % args.vol_desc)

    with open(args.vol_desc, 'r') as fp:
        data = np.loadtxt(fp, dtype=np.float, delimiter=',')

    assert len(data.shape) == 2
    assert data.shape[1] == 4
    n = data.shape[0]
    max_radius, max_volume = 1.75, 3.8**3

    with open(args.out_pml, "w+") as fp:
        fp.write("load %s\n" % args.pdb_file)
        fp.write("hide all\n")
        fp.write("bg_color white\n")
        for i in range(n):
            scale = max_radius * ((data[i, 3]/max_volume)**0.333)
            fp.write("pseudoatom g%d, pos=[%.3f, %.3f, %.3f]\n" % (i, data[i, 0], data[i, 1], data[i, 2]))
            fp.write("set sphere_scale, %.2f, g%d\n" % (scale, i))
            fp.write("show sphere, g%d\n" % i)
        fp.write("set sphere_transparency, 0.5\n")
        fp.write("select full_protein, chain %s\n" % args.chain)
        fp.write("show ribbon, full_protein\n")
        fp.write("set ribbon_color, red\n")
        fp.write("set ribbon_width, 5\n")
        fp.write("select target_res, resi %d and chain %s\n" % (args.residue, args.chain))
        fp.write("show sticks, target_res\n")
        fp.write("zoom resi %d\n" % args.residue)




