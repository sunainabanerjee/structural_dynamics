import os
import argparse
import numpy as np
import pandas as pd
import coarse_graining as cg
from matplotlib import pyplot as plt

__version__ = "1.0"
__all__ = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot validate sasa coarse grained model prediction accuracy")

    parser.add_argument('--feature-file', action='store', dest='feature_file',
                        type=str, required=True,
                        help="CSV file with features and actual sasa value")

    parser.add_argument('--model-file', action='store', dest='model_file',
                        type=str, required=True,
                        help="Model file containing the sasa model weights and definition")

    parser.add_argument('--model-type', action='store', dest='model_type',
                        type=str, choices=['mlp', 'xgb'], required=True,
                        help="Sasa model type")

    parser.add_argument('--n-points', action='store', dest="n_pts",
                        type=int, required=False, default=1000,
                        help="Number of points to use in the prediction test (default: 1000)")

    parser.add_argument('--out-file', action='store', dest='out_file',
                        type=str, required=True,
                        help="Out put file where the actual and prediction sasa score will be written")

    parser.add_argument('--to-plot', action='store_true', dest='to_plot',
                        required=False, default=False,
                        help="Tag to switch on the plot")

    args = parser.parse_args()

    if not os.path.isfile(args.model_file):
        raise Exception("Missing model file [%s]" % args.model_file)

    if not os.path.isfile(args.feature_file):
        raise Exception("Missing feature file [%s]" % args.feature_file)

    if not os.path.isdir(os.path.dirname(args.out_file)):
        raise Exception("Output directory missing [%s]" % os.path.dirname(args.out_file))

    if args.n_pts < 5:
        raise Exception("Not enough points to plot")

    with open(args.feature_file, "r") as fp:
        data = np.loadtxt(fp, delimiter=',')

    n, w = data.shape
    x = data[:, :-1]
    y = data[:, -1]

    model = None
    if args.model_type == 'mlp':
        model = cg.MLP()
    elif args.model_type == 'xgb':
        model = cg.XGBoost()
    model.load(args.model_file)

    idx = list(range(n))
    np.random.shuffle(idx)
    idx = idx[:args.n_pts]
    actual = y[idx].reshape(args.n_pts)
    pred = model.predict(x[idx, :]).reshape(args.n_pts)
    df = pd.DataFrame.from_dict({'actual': actual, 'predicted':pred})
    df.to_csv(args.out_file, index=False, header=True)
    if args.to_plot is True:
        plt.plot(actual, pred, '.r')
        plt.xlabel("actual")
        plt.ylabel("predicted")
        plt.show()
