import os
import time
import logging
import numpy as np
from geometry import *
from mutant_model import *
from structural_dynamics import *


if __name__ == "__main__":
    logging.basicConfig(debug=logging.DEBUG)
    ref_pdb_file = os.path.join(os.path.dirname(__file__), 'data', 'structure_1a1v.pdb')
    assert os.path.isfile(ref_pdb_file)
    ref_trace = pdb_to_catrace(read_pdb(pdb_file=ref_pdb_file)[0]['A'])
    max_crd, min_crd = Coordinate3d(31.559, 47.722, 38.290), \
                       Coordinate3d(2.412, 16.718, 18.526)
    buffer = 5.0
    for i in range(len(max_crd)):
        max_crd[i] += buffer
        min_crd[i] -= buffer

    top_dir = "/home/sumanta/Project/structural_dynamics/coarsegrained/martini/scripts/augmentation/nma"
    selected_directories = []
    assert os.path.isdir(top_dir)

    for d in os.listdir(top_dir):
        if os.path.isdir( os.path.join(top_dir, d)):
            sub_dir = os.path.join(top_dir, d)
            for m in os.listdir(sub_dir):
                if os.path.isdir(os.path.join(sub_dir, m)):
                    start_dir = os.path.join(sub_dir, m)
                    if os.path.isdir(os.path.join(start_dir, 'wild')) \
                            and \
                            os.path.isdir(os.path.join(start_dir, 'mutant')):
                        selected_directories.append(start_dir)

    np.random.shuffle(selected_directories)
    start = time.time()
    sig_gen = BulkSignatureGenerator()
    end = time.time()
    load_time = (end - start)
    print("Model Loading time : [%.2f s]" % load_time)

    modes = list(range(7, 25))
    fmt_string = "mode_catraj_{}.pdb"

    out_file = os.path.join( os.path.dirname(__file__), "out", "all_sig_bulk.csv")
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
        tag = [f for f in os.listdir(run_dir) if len(f.split("_")) == 4]
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
            mut_trj = align_trajectory(ref_trace, mut_trj[0], mut_trj, ca_trace=True)
            wild_trj = align_trajectory(ref_trace, wild_trj[0], wild_trj, ca_trace=True)
            in_end = time.time()

            print("Trajectory alignment time: [%.5f ms]" % ((in_end - in_start)*1000))

            in_start = time.time()
            wild_handler.add_trajectory(wild_trj)
            mutant_handler.add_trajectory(mut_trj)
            in_end = time.time()

            print("Trajectory addition time: [%.4f s]" % (in_end - in_start))
        end = time.time()
        print("Model preparation time : [%.2f s]" % (end - start))
        start = time.time()
        score = sig_gen.signature(wild_handler=wild_handler, mutant_handler=mutant_handler)
        end = time.time()
        print("Signature generation time: [%.2f s]" % (end - start))
        score = tag + ["%.5f" % f for f in score]
        with open(out_file, "a+") as fh:
            fh.write("%s\n" % ",".join(score))
        completed_list.append(tag[0])


