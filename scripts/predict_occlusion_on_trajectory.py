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

    args = parser.parse_args()

    if not os.path.isfile(args.trajectory):
        raise Exception("Error accessing trajectory file [%s]" % args.trajectory)

    if not os.path.isdir(args.model_folder):
        raise Exception("Error can not access model folder [%s]" % args.model_folder)

    trajectory = sd.read_trajectory_catrace(args.trajectory)
    assert len(trajectory) > 1
    trajectory = [inst[args.chain] for inst in trajectory]

    vol_predictor = cg.VolumePredictor(model_folder=args.model_folder,
                                       model_type=args.model_type)

    assert len(trajectory) > 1
    valid_residues = trajectory[0].residue_ids[1:-1]
    residues = valid_residues if len(args.residues) == 0 else args.residues
    assert all([r in valid_residues for r in residues])

    grid_predicted, vol_predicted, occ_predicted = dict(), dict(), dict()
    residue_names = dict()
    max_x, max_y, max_z = dict(), dict(), dict()
    min_x, min_y, min_z = dict(), dict(), dict()
    for i, pdb in enumerate(trajectory):
        print("Enumerating instance (%d)" % i)
        recons_unit = vol_predictor.predict_volume(pdb,
                                                   residues)
        for res_id, coordinates, occupancy in recons_unit:
            residue_names[res_id] = pdb.get_amino(res_id).name(one_letter_code=False)
            x, y, z = pdb.xyz(res_id)
            if res_id not in max_x:
                max_x[res_id] = min_x[res_id] = x
                max_y[res_id] = min_y[res_id] = y
                max_z[res_id] = min_z[res_id] = z

                grid_predicted[res_id] = list()
                vol_predicted[res_id] = list()
                occ_predicted[res_id] = list()

            if min_x[res_id] > x:
                min_x[res_id] = x
            if min_y[res_id] > y:
                min_y[res_id] = y
            if min_z[res_id] > z:
                min_z[res_id] = z
            if max_x[res_id] < x:
                max_x[res_id] = x
            if max_y[res_id] < y:
                max_y[res_id] = y
            if max_z[res_id] < z:
                max_z[res_id] = z

            grid_predicted[res_id].append(coordinates)
            occ_predicted[res_id].append(occupancy)
            vol_predicted[res_id].append(np.sum(occupancy))

    residues = grid_predicted.keys()
    gx, gy, gz = cg.VolumeSignatureGrid.size()
    sx, sy, sz = cg.VolumeSignatureGrid.dim()
    buffer_x, buffer_y, buffer_z = gx * ((sx + 1) // 2), \
                                   gy * ((sy + 1) // 2), \
                                   gz * ((sz + 1) // 2)
    for res_id in residues:
        mx_coord = Coordinate3d(max_x[res_id] + buffer_x,
                                max_y[res_id] + buffer_y,
                                max_z[res_id] + buffer_z)
        mn_coord = Coordinate3d(min_x[res_id] - buffer_x,
                                min_y[res_id] - buffer_y,
                                min_z[res_id] - buffer_z)
        residue_volume = list()
        lookup_grid = LookupGrid(min_crd=mn_coord,
                                 max_crd=mx_coord,
                                 size=3.8,
                                 max_cutoff=3.8 ** 3)
        for i, grid in enumerate(grid_predicted[res_id]):
            lookup_grid.reset()
            n, __ = grid.shape
            for j in range(n):
                assert lookup_grid.inside(*grid[j, :])
                lookup_grid[grid[j, :]] += occ_predicted[res_id][i][j]
            residue_volume.append([lookup_grid[i] for i in range(lookup_grid.size)])
        avg_vol = np.mean(vol_predicted[res_id])
        occ_vol = np.sum(np.max(np.array(residue_volume), axis=0))
        print('%d\t%s\t%.3f\t%.3f' % (res_id, residue_names[res_id], avg_vol, occ_vol))


