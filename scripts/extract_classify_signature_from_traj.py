import os
import time
import logging
import argparse
import numpy as np
from geometry import *
from mutant_model import *
from structural_dynamics import *
import classifier as cls

__version__ = '1.0'


def read_bounding_box( filename ):
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
    assert all([max_coordinate[i] > min_coordinate[i] for i in range(len(max_coordinate))])
    return max_coordinate, min_coordinate


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

    logger = logging.getLogger('MAIN')
    parser = argparse.ArgumentParser(description="Utility builds the input for sasa prediction")

    parser.add_argument('--ref-pdb', action='store', dest='ref_pdb',
                        help="reference pdb used to align the trajectory to box definition",
                        type=str, required=True)

    parser.add_argument('--box', action='store', dest='box_defn',
                        help="box defined in axis parallel way, by min and max coordinates",
                        type=str, required=True)

    parser.add_argument('--trajectory', action='store', dest='traj_dir',
                        help="directory containing nma trajectory of the wild and mutant",
                        type=str, required=True)

    parser.add_argument('--buffer', action='store', dest='buffer', default=2.0,
                        help='buffer distance for grid build',
                        type=float, required=False)

    parser.add_argument('--alignment', action='store', dest='align_map',
                        help="file containing residue correspondence to be used "
                             "for the alignment with the reference pdb",
                        type=str, required=False)

    parser.add_argument('--model', action='store', dest='model',
                        type=str, required=True,
                        help="xgboost model file for classification")

    args = parser.parse_args()
    assert os.path.isfile(args.ref_pdb)
    ref_pdb_file = os.path.abspath(args.ref_pdb)

    assert os.path.isdir(args.traj_dir)
    assert os.path.isdir(os.path.join(args.traj_dir, 'wild'))
    assert os.path.isdir(os.path.join(args.traj_dir, 'mutant'))
    assert os.path.isfile(args.model)
    trj_dir = os.path.abspath(args.traj_dir)

    ref_list = read_pdb(pdb_file=ref_pdb_file)
    assert isinstance(ref_list, list) and len(ref_list) == 1
    chains = list(ref_list[0].keys())
    assert len(chains) == 1
    ref_trace = pdb_to_catrace(ref_list[0][chains[0]])

    assert os.path.isfile(args.box_defn)
    max_crd, min_crd = read_bounding_box(args.box_defn)
    for i in range(len(max_crd)):
        max_crd[i] += args.buffer
        min_crd[i] -= args.buffer

    align_map = None
    if args.align_map is not None:
        align_map = parse_alignment(args.align_map)

    wild_traj_dir = os.path.join(trj_dir, 'wild')
    mutant_traj_dir = os.path.join(trj_dir, 'mutant')

    start = time.time()
    sig_gen = BulkSignatureGenerator()
    end = time.time()
    logger.debug("Model loading time: [%.3f s]" % (end - start))
    print("Model loading time: [%.3f s]" % (end - start))

    wild_handler = NMATrajectoryHandler(min_coordinate=min_crd, max_coordinate=max_crd)
    mutant_handler = NMATrajectoryHandler(min_coordinate=min_crd, max_coordinate=max_crd)
    modes = list(range(7, 25))
    start = time.time()
    for mode in modes:
        wild_handler.reset()
        mutant_handler.reset()
        wild_trj_file = os.path.join(wild_traj_dir, 'mode_catraj_%d.pdb' % mode)
        mutant_trj_file = os.path.join(mutant_traj_dir, 'mode_catraj_%d.pdb' % mode)

        assert os.path.isfile(wild_trj_file) and os.path.isfile(mutant_trj_file)
        wild_traj = [extract_trace(pair) for pair in read_trajectory_catrace(wild_trj_file)]
        mutant_traj = [extract_trace(pair) for pair in read_trajectory_catrace(mutant_trj_file)]

        in_start = time.time()
        mutant_traj = align_trajectory(ref_trace, mutant_traj[0], mutant_traj, align_pair=align_map, ca_trace=True)
        wild_traj = align_trajectory(ref_trace, wild_traj[0], wild_traj, align_pair=align_map, ca_trace=True)
        in_end = time.time()
        logger.debug("Trajectory alignment time: [%.4f s]" % (in_end - in_start))
        print("Trajectory alignment time: [%.4f s]" % (in_end - in_start))

        in_start = time.time()
        mutant_handler.add_trajectory(mutant_traj)
        wild_handler.add_trajectory(wild_traj)
        in_end = time.time()
        logger.debug("Trajectory addition time: [%.4f s]" % (in_end - in_start))
        print("Trajectory addition time: [%.4f s]" % (in_end - in_start))
    end = time.time()
    logger.debug("Model preparation time : [%.2f s]" % (end - start))
    print("Model preparation time : [%.2f s]" % (end - start))

    start = time.time()
    score = sig_gen.signature(wild_handler=wild_handler, mutant_handler=mutant_handler)
    end = time.time()
    logger.debug("Signature generation time: [%.2f s]" % (end - start))

    model = cls.XGBoost(booster=cls.XGBooster.gblinear(),
                        max_depth=4,
                        n_estimators=300,
                        n_class=3)
    model.load(args.model)
    score = np.array(score).reshape((1, len(score)))
    predicted_proba = model.predict_proba(score)[0]
    predicted_class = model.predict(score)
    print("Class Scores: (%.3f, %.3f, %.3f)" % (predicted_proba[0],
                                                predicted_proba[1],
                                                predicted_proba[2]))
    print("Class Label: %d" % predicted_class)




