import os
import argparse
import coarse_graining as cg
import structural_dynamics as sd
from matplotlib import pyplot as plt

__version__ = "1.0"
__all__ = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot validate sasa coarse grained model prediction accuracy")

    parser.add_argument('--pdb-file', action='store', dest='pdb_file',
                        type=str, required=True,
                        help="target pdb file against which the prediction will be performed")

    parser.add_argument('--model-folder', action='store', dest='model_folder',
                        type=str, required=True,
                        help="Model file containing the sasa model weights and definition")

    parser.add_argument('--model-type', action='store', dest='model_type',
                        type=str, choices=['mlp', 'xgb'], required=False,
                        default='xgb',
                        help="Sasa model type")

    parser.add_argument('--to-plot', action='store_true', dest='to_plot',
                        required=False, default=False,
                        help="Tag to switch on the plot")

    args = parser.parse_args()

    if not os.path.isdir(args.model_folder):
        raise Exception("Missing model file [%s]" % args.model_file)

    if not os.path.isfile(args.pdb_file):
        raise Exception("Missing feature file [%s]" % args.pdb_file)

    pdb_list = sd.read_pdb(args.pdb_file)
    sasa_predictor = cg.SASAPredictor(model_folder=args.model_folder, model_type=args.model_type)

    for pdbs in pdb_list:
        assert isinstance(pdbs, dict)
        chains = list(pdbs.keys())
        for chain in chains:
            ca_trace = sd.pdb_to_catrace(pdbs[chain])
            residues, sasa = sasa_predictor.predict_sasa(ca_trace, ca_trace.residue_ids)
            assert len(residues) == len(sasa)
            for i, res in enumerate(residues):
                print("%d\t%s\t%.2f" % (res, ca_trace.get_amino(res), sasa[i]))

            if args.to_plot is True:
                plt.plot(residues, sasa, c='r')
                plt.xlabel("residues")
                plt.ylabel("SASA")
                plt.show()
