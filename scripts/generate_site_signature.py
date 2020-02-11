import os
import time
import logging
import argparse
from geometry import *
from mutant_model import *
from structural_dynamics import *

__version__ = "1.0"


def read_bounding_box(filename):
    assert os.path.isfile(filename)
    min_coordinate, max_coordinate = None, None
    with open(filename, "r") as fp:
        for line in fp.readlines():
            if line.count(',') == 0:
                continue
            flds = line.split(',')
            if flds[0] == 'min':
                assert len(flds) == 4
                min_coordinate = Coordinate3d(float(flds[1]), float(flds[2]), float(flds[3]))
            elif flds[0] == 'max':
                assert len(flds) == 4
                max_coordinate = Coordinate3d(float(flds[1]), float(flds[2]), float(flds[3]))
    assert (min_coordinate is not None) and (max_coordinate is not None)
    return min_coordinate, max_coordinate


def extract_trace( pair ):
    assert isinstance(pair, dict)
    assert len(pair.keys()) == 1
    chains = list(pair.keys())
    return pair[chains[0]]


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
    logging.basicConfig(debug=logging.DEBUG)
    parser = argparse.ArgumentParser(description="Script generates bulk signature for multiple pdbs.")
    parser.add_argument('--ref-pdb', action='store', dest='ref_pdb_file',
                        required=True, type=str,
                        help="PDB/CaTrace containing reference pdb against which binding site location is defined.")
    parser.add_argument('--box', action='store', dest='box_defn',
                        type=str, required=True,
                        help="CSV file describes the binding site box")
    parser.add_argument('--top-dir', action='store', dest='top_dir',
                        required=True, type=str,
                        help="Top directory where all the NMA trajectories")
    parser.add_argument('--alignment', action='store', dest='align_map',
                        help="file containing residue correspondence to be used "
                             "for the alignment with the reference pdb",
                        type=str, required=False)
    parser.add_argument('--out', type=str, action='store', required=True,
                        dest='out_file', help="Output file where the result will be appended")
    parser.add_argument('--buffer', type=float, action='store', dest='buffer',
                        required=False, default=2.0,
                        help="Buffer length to define larger box than the box described,"
                             "(default : 2.0)")

    args = parser.parse_args()

    ref_pdb_file = args.ref_pdb_file
    assert os.path.isfile(ref_pdb_file)
    pdbs = read_pdb(pdb_file=ref_pdb_file)
    assert isinstance(pdbs, list) and len(pdbs) == 1
    assert len(pdbs[0].keys()) == 1
    chain = list(pdbs[0].keys())[0]
    ref_trace = pdbs[0][chain]
    if isinstance(ref_trace, PDBStructure):
        ref_trace = pdb_to_catrace(ref_trace)
    assert isinstance(ref_trace, CaTrace)

    assert os.path.isfile(args.box_defn)
    min_crd, max_crd = read_bounding_box(args.box_defn)

    assert args.buffer >= 0.0
    buffer = args.buffer
    for i in range(len(max_crd)):
        max_crd[i] += buffer
        min_crd[i] -= buffer

    top_dir = args.top_dir
    selected_directories = []
    assert os.path.isdir(top_dir)

    assert os.path.isdir(os.path.join(top_dir, 'wild'))
    assert os.path.isdir(os.path.join(top_dir, 'mutant'))
    selected_directories = [top_dir]

    align_map = None
    if args.align_map is not None:
        align_map = parse_alignment(args.align_map)

    start = time.time()
    sig_gen = BulkSignatureGenerator()
    end = time.time()
    load_time = (end - start)
    print("Model Loading time : [%.2f s]" % load_time)

    modes = list(range(7, 25))
    fmt_string = "mode_catraj_{}.pdb"

    assert os.path.isdir(os.path.dirname(args.out_file))
    out_file = args.out_file
    completed_list = []
    if os.path.isfile(out_file):
        with open(out_file, "r+") as fp:
            completed_list = [line.split(",")[0] for line in fp.readlines()]

    wild_handler = NMATrajectoryHandler(min_coordinate=min_crd, max_coordinate=max_crd)
    mutant_handler = NMATrajectoryHandler(min_coordinate=min_crd, max_coordinate=max_crd)
    for run_dir in selected_directories:
        print("Current number of complete list [%d] " % len(completed_list))
        wild_handler.reset()
        mutant_handler.reset()
        tag = [f for f in os.listdir(run_dir) if f.count("_") > 0]
        assert len(tag) == 1
        if tag[0] in completed_list:
            continue
        print("Processing: %s" % tag[0])
        wild_dir = os.path.join(run_dir, "wild")
        mutant_dir = os.path.join(run_dir, "mutant")
        assert os.path.isdir(wild_dir) and os.path.isdir(mutant_dir)
        start = time.time()
        for m in modes:
            print("Accessing mode [%d]" % m)
            wild_pdb = os.path.join(wild_dir, fmt_string.format(m))
            mutant_pdb = os.path.join(mutant_dir, fmt_string.format(m))

            assert os.path.isfile(wild_pdb) and os.path.isfile(mutant_pdb)
            mut_trj = [pair['A'] for pair in read_trajectory_catrace(mutant_pdb)]
            wild_trj = [pair['A'] for pair in read_trajectory_catrace(wild_pdb)]

            in_start = time.time()
            mut_trj = align_trajectory(ref_trace, mut_trj[0], mut_trj, align_pair=align_map, ca_trace=True)
            wild_trj = align_trajectory(ref_trace, wild_trj[0], wild_trj, align_pair=align_map, ca_trace=True)
            in_end = time.time()

            print("Trajectory alignment time: [%.4f s]" % (in_end - in_start))

            in_start = time.time()
            wild_handler.add_trajectory(wild_trj)
            mutant_handler.add_trajectory(mut_trj)
            in_end = time.time()

            print("Trajectory addition time: [%.4f s]" % (in_end - in_start))
        end = time.time()
        print("Model preparation time : [%.4f s]" % (end - start))
        start = time.time()
        score = sig_gen.signature(wild_handler=wild_handler, mutant_handler=mutant_handler)
        end = time.time()
        print("Signature generation time: [%.4f s]" % (end - start))
        score = tag + ["%.5f" % f for f in score]
        with open(out_file, "a+") as fh:
            fh.write("%s\n" % ",".join(score))
        completed_list.append(tag[0])


