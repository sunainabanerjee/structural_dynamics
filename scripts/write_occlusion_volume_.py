import os
import argparse
import numpy as np
from geometry import *
from mutant_model import *
import coarse_graining as cg
import structural_dynamics as sd

__version__ = "1.0"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot validate occlusion coarse grained "
                                                 "model prediction accuracy. Handles only "
                                                 "one chain at a time")

    parser.add_argument('--trajectory', action='store', dest='trajectory',
                        type=str, required=True,
                        help="target trajectory file against which the prediction "
                             "will be performed")

    parser.add_argument('--chain', action='store', dest='chain',
                        type=str, required=False, default='A',
                        help='chain of the target structure, default chain ID [A]')

    parser.add_argument('--model-folder', action='store', dest='model_folder',
                        type=str, required=True,
                        help="Model file containing the sasa model weights and definition")

    parser.add_argument('--model-type', action='store', dest='model_type',
                        type=str, choices=['mlp', 'xgb'], required=False,
                        default='mlp', help="Sasa model type")

    parser.add_argument('--residue', nargs='+', dest='residues',
                        required=False, default=list(), type=int,
                        help="residues on which the prediction needs to be performed! "
                             "Default behaviour considers all residues")

    parser.add_argument('--occlusion-pdb', action='store', dest='out_file',
                        required=True, type=str,
                        help="Outpdb where the occlusion volume will be written")

    args = parser.parse_args()

    if not os.path.isfile(args.trajectory):
        raise Exception("Error accessing trajectory file [%s]" % args.trajectory)

    if not os.path.isdir(args.model_folder):
        raise Exception("Error can not access model folder [%s]" % args.model_folder)

    trajectory = sd.read_trajectory_catrace(args.trajectory)
    assert len(trajectory) > 0
    trajectory = [inst[args.chain] for inst in trajectory]

    vol_predictor = cg.VolumePredictor(model_folder=args.model_folder,
                                       model_type=args.model_type)

    assert len(trajectory) > 0
    valid_residues = trajectory[0].residue_ids[1:-1]
    residues = valid_residues if len(args.residues) == 0 else args.residues
    assert all([r in valid_residues for r in residues])

    grid_predicted, occ_predicted = dict(), dict()
    for i, pdb in enumerate(trajectory):
        print("Enumerating instance (%d)" % i)
        recons_unit = vol_predictor.predict_volume(pdb,
                                                   residues)
        for res_id, coordinates, occupancy in recons_unit:
            if res_id not in grid_predicted:
                grid_predicted[res_id] = list()
                occ_predicted[res_id] = list()

            grid_predicted[res_id].append(coordinates)
            occ_predicted[res_id].append(occupancy)

    index = 0
    fh = open(args.out_file, 'w+')
    for res_id in grid_predicted.keys():
        for i, occ_grid in enumerate(grid_predicted[res_id]):
            assert isinstance(occ_grid, np.ndarray)
            for j, occ in enumerate(occ_predicted[res_id][i]):
                index = index + 1
                fh.write("ATOM  %5d  %-3s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f \n" % (index,
                                                                                         'SPH',
                                                                                         'SPH',
                                                                                         'Z',
                                                                                         int(res_id),
                                                                                         float(occ_grid[j, 0]),
                                                                                         float(occ_grid[j, 1]),
                                                                                         float(occ_grid[j, 2]),
                                                                                         1.0,
                                                                                         float(occ)))
    fh.write("TER\n")
    fh.close()


