import os
import argparse
import coarse_graining as cg
import structural_dynamics as sd

__version__ = "1.0"
__all__ = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot validate volume coarse grained "
                                                 "model prediction accuracy. Handles "
                                                 "only one chain at a time")

    parser.add_argument('--pdb-file', action='store', dest='pdb_file',
                        type=str, required=True,
                        help="target pdb file against which the prediction will be performed")

    parser.add_argument('--chain', action='store', dest='chain',
                        type=str, required=False, default='A',
                        help='chain of the target structure, default chain ID [A]')

    parser.add_argument('--model-folder', action='store', dest='model_folder',
                        type=str, required=True,
                        help="Model file containing the sasa model weights and definition")

    parser.add_argument('--model-type', action='store', dest='model_type',
                        type=str, choices=['mlp', 'xgb'], required=False,
                        default='mlp',
                        help="Sasa model type")

    parser.add_argument('--residue', nargs='+', dest='residues',
                        required=False, default=list(), type=int,
                        help="residues on which the prediction needs to be performed! "
                             "Default behaviour considers all residues")

    parser.add_argument('--output-directory', action='store', dest='out_folder',
                        required=True, type=str,
                        help="Output folder location, where all the residue wise "
                             "values will be stored!")

    args = parser.parse_args()

    if not os.path.isfile(args.pdb_file):
        raise Exception("Not a valid PDB file [%s]" % args.pdb_file)

    if not os.path.isdir(args.model_folder):
        raise Exception("Not a valid model folder [%s]" % args.model_folder)

    if not os.path.isdir(args.out_folder):
        raise Exception("Out folder does not exists [%s]" % args.out_folder)

    pdb_list = sd.read_pdb(args.pdb_file)
    chain = args.chain
    residues = args.residues

    assert len(pdb_list) == 1

    vol_predictor = cg.VolumePredictor(model_folder=args.model_folder,
                                       model_type=args.model_type)

    for pdbs in pdb_list:
        assert isinstance(pdbs, dict)
        assert chain in pdbs
        ca_trace = sd.pdb_to_catrace(pdbs[chain])
        all_residues = ca_trace.residue_ids[1:-1]
        if len(residues) == 0:
            residues = all_residues
        assert all([r in all_residues for r in residues])
        recons_unit = vol_predictor.predict_volume(ca_trace,
                                                   residues)

        for resId, coordinates, occupancy in recons_unit:
            out_file = os.path.join(args.out_folder, 'occupancy_grid_%d.csv' % resId)
            assert coordinates.shape[0] == len(occupancy)
            with open(out_file, 'w+') as fp:
                for i, occ in enumerate(occupancy):
                    fp.write("%.3f,%.3f,%.3f,%.2f\n" % (coordinates[i, 0],
                                                        coordinates[i, 1],
                                                        coordinates[i, 2],
                                                        occ))

