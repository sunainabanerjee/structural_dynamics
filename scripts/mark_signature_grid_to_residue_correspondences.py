import os
import argparse
import numpy as np
import pandas as pd
from geometry import *
import mutant_model as mm
import structural_dynamics as sd

__version__ = "1.0"


def read_bounding_box(filename):
    assert os.path.isfile(filename)
    with open(filename, 'r+') as fp:
        lines = fp.readlines()
    assert len(lines) > 0
    max_coordinate, min_coordinate = None, None
    for line in lines:
        flds = line.split(",")
        if (len(flds) == 4) and (flds[0] == "max"):
            max_coordinate = Coordinate3d(float(flds[1]), float(flds[2]), float(flds[3]))
        if (len(flds) == 4) and (flds[0] == "min"):
            min_coordinate = Coordinate3d(float(flds[1]), float(flds[2]), float(flds[3]))
    assert max_coordinate is not None
    assert min_coordinate is not None
    assert all([max_coordinate[idx] > min_coordinate[idx] for idx in range(len(max_coordinate))])
    return max_coordinate, min_coordinate


def parse_alignment( file ):
    assert os.path.isfile(file)
    with open(file, 'r') as fp:
        lines = fp.readlines()
    assert len(lines) > 4
    residue_map = list()
    for line in lines:
        if line.count(',') == 1:
            residue_map.append(tuple([int(fld) for fld in line.split(',')]))
    assert len(residue_map) > 4
    return residue_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mark residue correspondence to "
                                                 "signature grid from reference "
                                                 "pdb and bounding box definition")

    parser.add_argument('--trajectory', action='store', dest='trajectory',
                        type=str, required=True,
                        help="target trajectory file against which the "
                             "prediction will be performed")

    parser.add_argument('--chain', action='store', dest='chain',
                        type=str, required=False, default='A',
                        help="chain of the target structure, default "
                             "chain ID [A]")

    parser.add_argument('--ref-pdb', action='store', dest='ref_pdb',
                        type=str, required=False, default=None,
                        help="Reference pdb to be used to align the trajectory "
                             "to bounding box. This must contain only a single "
                             "chain. If not provided it is assumed the trajectory "
                             "pdb is already aligned.")

    parser.add_argument('--residue-correspondence', action='store', dest='align_map',
                        type=str, required=False, default=None,
                        help="Residue correspondence to use while align target "
                             "trajectory to reference pdb. In case not mentioned, "
                             "it assumes two pdbs are sequence homologues and one "
                             "to one residue correspondences.")

    parser.add_argument('--bounding-box', action='store', dest='bb_file',
                        type=str, required=True,
                        help="Maximum and minimum coordinates that encapsulate "
                             "the site.")

    parser.add_argument('--buffer', action='store', dest='buffer', default=2.0,
                        help='buffer distance for grid build',
                        type=float, required=False)

    parser.add_argument('--residence-cutoff', action='store', dest='cutoff',
                        default=1.0, type=float, required=False,
                        help='Residence cutoff to list final residue sets')

    parser.add_argument('--table', action='store_true', dest='table',
                        default=False,
                        help='Output result in table format!')

    parser.add_argument('--out', action='store', dest='out_file',
                        required=True, type=str,
                        help='Output file name')

    args = parser.parse_args()

    if not os.path.isfile(args.trajectory):
        raise Exception("Error accessing trajectory file [%s]" % args.trajectory)

    if (args.ref_pdb is not None) and (not os.path.isfile(args.ref_pdb)):
        raise Exception("Error can not access reference pdb [%s]" % args.ref_pdb)

    if not os.path.isfile(args.bb_file):
        raise Exception("Error can not find the bounding box definition file")

    if (args.align_map is not None) and (not os.path.isfile(args.align_map)):
        raise Exception("Error invalid alignment map provided [%s]!" % args.align_map)

    assert (args.cutoff >= 0) and (args.cutoff <= 1.0)

    trajectory = sd.read_trajectory_catrace(args.trajectory)
    assert len(trajectory) > 0

    max_coordinate, min_coordinate = read_bounding_box(args.bb_file)
    for i in range(len(max_coordinate)):
        max_coordinate[i] += args.buffer
        min_coordinate[i] -= args.buffer

    trajectory = [snap[args.chain] for snap in trajectory]

    align_map = None
    if args.align_map is not None:
        align_map = parse_alignment(args.align_map)

    if args.ref_pdb is not None:
        ref_pdb = sd.read_pdb(args.ref_pdb)
        assert (len(ref_pdb) == 1) and (len(ref_pdb[0]) == 1)
        chain = list(ref_pdb[0].keys())[0]
        ref_pdb = sd.pdb_to_catrace(ref_pdb[0][chain])
        trajectory = mm.align_trajectory(ref_structure=ref_pdb,
                                         tgt_structure=trajectory[0],
                                         trajectory=trajectory,
                                         align_pair=align_map,
                                         ca_trace=True)

    signature = mm.Signature(max_coord=max_coordinate,
                             min_coord=min_coordinate)

    res_names = dict()
    for trace in trajectory:
        for res_id in trace.residue_ids:
            if res_id not in res_names:
                res_names[res_id] = trace.get_amino(res_id).name(one_letter_code=False)
            x, y, z = trace.xyz(res_id)
            signature.add(x, y, z, res_id)

    gx, gy, gz = signature.grid_dim
    counter = 0
    residues_corr = list()
    for ix in range(gx-1):
        for iy in range(gy-1):
            for iz in range(gz-1):
                residues_corr.append(((ix, iy, iz), np.array(signature[counter])))
                counter += 1

    n = len(trajectory)
    if args.table:
        all_hits = set()
        all_grids = list()
        for g_idx, residues in residues_corr:
            all_grids.append('%d-%d-%d' % (g_idx[0]+1, g_idx[1]+1, g_idx[2]+1))
            for r in residues:
                name = '%s%d' % (res_names[r], r)
                all_hits.add(name)
        all_hits = {hits: [] for hits in all_hits}
        for g_idx, residues in residues_corr:
            res_freq = {'%s%d' % (res_names[r], r): len(np.where(residues == r)[0])/n for r in np.unique(residues)}
            for hits in all_hits:
                score = res_freq[hits] if hits in res_freq else 0
                all_hits[hits].append(score)
        result = pd.DataFrame(all_hits, index=all_grids)
        result.to_csv(args.out_file, sep=',', index=True, header=True)
    else:
        with open(args.out_file, 'w+') as fp:
            for g_idx, residues in residues_corr:
                res_freq = {'%s%d' % (res_names[r], r): len(np.where(residues == r)[0])/n for r in np.unique(residues)}
                res_hits = [r for r in res_freq if res_freq[r] >= args.cutoff]
                fp.write("%d-%d-%d,%s\n" % (g_idx[0]+1, g_idx[1]+1, g_idx[2]+1, ":".join(res_hits)))
