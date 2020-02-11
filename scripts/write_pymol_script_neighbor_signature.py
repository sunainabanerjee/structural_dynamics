import os
import argparse
from geometry import *
import coarse_graining as cg
import structural_dynamics as sd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Produce pymol script for a volume visualization")

    parser.add_argument('--pdb-file', action='store', dest='pdb_file',
                        type=str, required=True,
                        help="target pdb file against which the prediction will be performed")

    parser.add_argument("--chain", action='store', dest='chain',
                        type=str, required=False, default='A',
                        help="PDB chain id")

    parser.add_argument("--residue", action='store', dest='residue',
                        type=int, required=True,
                        help="Residue Id of the target protein for the given description file")

    parser.add_argument("--topk", action='store', dest='topk',
                        type=int, required=False, default=4,
                        help="Number of neighboring residues (default: 4)")

    parser.add_argument("--out", action='store', dest='out_pml',
                        type=str, required=True,
                        help="Out file location")

    args = parser.parse_args()

    if not os.path.isfile(args.pdb_file):
        raise Exception("Can not read pdb file [%s]" % args.pdb_file)

    pdb_list = sd.read_pdb(args.pdb_file)
    chain = args.chain

    assert len(pdb_list) == 1
    neigbors = []
    r_0, r_1, r_2 = args.residue - 1, args.residue, args.residue + 1

    for pdbs in pdb_list:
        assert isinstance(pdbs, dict)
        assert chain in pdbs
        ca_trace = sd.pdb_to_catrace(pdbs[chain])
        valid_residues = ca_trace.residue_ids[1:-1]
        if args.residue not in valid_residues:
            raise Exception("Residue %d not a valid residue!" % args.residue)
        neighbors = cg.find_k_neighbors(pdb=ca_trace, residues=[args.residue], k=args.topk + 2)
        neighbors = [r for r, d in neighbors[args.residue] if abs(r - args.residue) > 1]
        volumes = [cg.PropertyCalculator.volume(ca_trace, r) for r in neighbors]
        hydropathy = [cg.PropertyCalculator.hydropathy(ca_trace, r) for r in neighbors]
        idx = ca_trace.residue_ids.index(args.residue)
        all_residues = ca_trace.residue_ids
        r_0, r_1, r_2 = all_residues[idx - 1], all_residues[idx], all_residues[idx + 1]
        ca_prev, ca_curr, ca_next = Coordinate3d(*ca_trace.xyz(r_0)), \
                                    Coordinate3d(*ca_trace.xyz(r_1)), \
                                    Coordinate3d(*ca_trace.xyz(r_2))
        uv = (connecting_vector(ca_prev, ca_curr) + connecting_vector(ca_next, ca_curr)).unit_vector
        d = 0.5*(connecting_vector(ca_prev, ca_curr).norm + connecting_vector(ca_next, ca_curr).norm)
        coord = ca_curr + d * uv

    max_volume, min_volume = 227, 60
    max_hydropathy, min_hydropathy = 4.5, -4.5
    hydropathy_colors = ['helium', 'fluorine', 'lanthanum', 'krypton', 'molybdenum', 'rhodium' ]

    with open(args.out_pml, 'w+') as fp:
        fp.write("load %s\n" % args.pdb_file)
        fp.write("hide all\n")
        fp.write("select protein_tgt, chain %s\n" % chain)
        fp.write("select ca0, chain %s and resi %d and name CA\n" % (args.chain, r_0))
        fp.write("select ca1, chain %s and resi %d and name CA\n" % (args.chain, r_1))
        fp.write("select ca2, chain %s and resi %d and name CA\n" % (args.chain, r_2))
        fp.write("pseudoatom ref, pos=[%.3f, %.3f, %.3f]\n" % (coord[0], coord[1], coord[2]))
        fp.write("show ribbon, protein_tgt\n")
        fp.write("set ribbon_color, indium\n")
        fp.write("show surface, protein_tgt\n")
        fp.write("set surface_color, hydrogen\n")
        fp.write("set transparency, 0.8\n")
        fp.write("set dash_gap, 0\n")
        fp.write("distance pre_vector, /ca0, /ca1, label=0\n")
        fp.write("distance post_vector, /ca1, /ca2, label=0\n")
        fp.write("distance ref_vector, /ca1, /ref, label=0\n")
        fp.write("color red, pre_vector\n")
        fp.write("color red, post_vector\n")
        fp.write("color red, ref_vector\n")
        fp.write("show sphere, ca0\n")
        fp.write("show sphere, ca1\n")
        fp.write("show sphere, ca2\n")
        fp.write("set sphere_scale, 0.5, ca0\n")
        fp.write("set sphere_color, cobalt, ca0\n")
        fp.write("set sphere_scale, 0.5, ca1\n")
        fp.write("set sphere_color, oxygen, ca1\n")
        fp.write("set sphere_scale, 0.5, ca2\n")
        fp.write("set sphere_color, cobalt, ca2\n")
        fp.write("bg_color white\n")
        for i, nbr in enumerate(neighbors):
            volume_scale = ((volumes[i] - min_volume) / (max_volume - min_volume)) + 0.5
            hydropathy_scale = (hydropathy[i] - min_hydropathy) / (max_hydropathy - min_hydropathy)
            hcolor = hydropathy_colors[int(hydropathy_scale / 0.2)]
            fp.write("select nbr%d, chain %s and resi %d and name CA\n" % (i, args.chain, nbr))
            fp.write("show sphere, nbr%d\n" % i)
            fp.write("set sphere_scale, %.2f, nbr%d\n" % (volume_scale, i))
            fp.write("set sphere_transparency, 0.3, nbr%d\n" % i)
            fp.write("set sphere_color, %s, nbr%d\n" % (hcolor, i))
            fp.write("distance nbr_connect_%d, /ca1, /nbr%d, label=0\n" % (i, i))
            fp.write("color molybdenum, nbr_connect_%d\n" % i)
            fp.write('label nbr%d, "Nbr%d"\n' % (i, i+1))
        fp.write("angle ca_nbr0, (nbr0), (ca1), (ref), label=0\n")
        fp.write("set angle_color, nobelium\n")
        fp.write('label ref, u"\\u03b8".encode("utf-8")\n')
        fp.write('hide lines\n')
        fp.write('set ribbon_width, 0.5\n')
        fp.write('label ca0, u"C\\u03b1-1".encode("utf-8")\n')
        fp.write('label ca1, u"C\\u03b1".encode("utf-8")\n')
        fp.write('label ca2, u"C\\u03b1+1".encode("utf-8")\n')








