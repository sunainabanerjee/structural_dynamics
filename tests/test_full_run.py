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
    sig_gen = SignatureGenerator(min_crd, max_crd)
    end = time.time()
    load_time = (end - start)
    print("Model Loading time : [%.2f s]" % load_time)

    modes = list(range(7, 10))
    fmt_string = "mode_catraj_{}.pdb"

    out_file = os.path.join( os.path.dirname(__file__), "out", "all_sig.csv")

    for run_dir in selected_directories:
        tag = [f for f in os.listdir(run_dir) if len(f.split("_")) == 4]
        assert len(tag) == 1
        print(tag)
        wild_dir = os.path.join(run_dir, "wild")
        mutant_dir = os.path.join(run_dir, "mutant")
        assert os.path.isdir(wild_dir) and os.path.isdir(mutant_dir)
        tot_sig_m = {p: None for p in sig_gen.assigned_properties}
        tot_sig_w = {p: None for p in sig_gen.assigned_properties}
        for m in modes:
            wild_pdb = os.path.join(wild_dir, fmt_string.format(m))
            mutant_pdb = os.path.join(mutant_dir, fmt_string.format(m))

            assert os.path.isfile(wild_pdb) and os.path.isfile(mutant_pdb)
            mut_trj = [pair['A'] for pair in read_trajectory_catrace(mutant_pdb)]
            wild_trj = [pair['A'] for pair in read_trajectory_catrace(wild_pdb)]

            msig = trajectory_signature(ref_trace, mut_trj, sig_gen)
            wsig = trajectory_signature(ref_trace, wild_trj, sig_gen)
            for p in msig.keys():
                if tot_sig_m[p] is None:
                    tot_sig_m[p] = msig[p]
                    tot_sig_w[p] = wsig[p]
                else:
                    tot_sig_m[p] = merge_signature(tot_sig_m[p],
                                                   msig[p],
                                                   operation=SignatureAggregationOp.get_operation(p))
                    tot_sig_w[p] = merge_signature(tot_sig_w[p],
                                                   wsig[p],
                                                   operation=SignatureAggregationOp.get_operation(p))
        fscore = []
        for p in sig_gen.assigned_properties:
            sscore = signature_diff(tot_sig_m[p], tot_sig_w[p])
            fscore = fscore + [m for m, s in sscore] + [s for m, s in sscore]
        fscore = tag + ["%.5f" % f for f in fscore]
        with open(out_file, "a+") as fh:
            fh.write("%s\n" % ",".join(fscore))


