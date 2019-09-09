import os
import logging
import argparse
import numpy as np
import regressor as reg
import coarse_graining as cg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to train the sasa models from feature csv")

    parser.add_argument('--in', action='store', dest='in_csv',
                        help="input csv, containing the feature description and regressor value",
                        type=str, required=True)

    parser.add_argument('--nepoch', action='store', dest='nepoch',
                        default=2500,
                        help='number of epochs in training (default: 2500)',
                        type=int, required=False)

    parser.add_argument('--batch-size', action='store', dest='batch_size',
                        default=100,
                        help="number of data each batch (default: 100)",
                        type=int, required=False)

    parser.add_argument('--model-type', action='store', dest='model_type',
                        choices=['xgb', 'mlp'],
                        type=str,
                        default='mlp',
                        help="specify the model type (default: mlp)",
                        required=False)

    parser.add_argument('--model-out', action='store', dest='model_out',
                        type = str, required=True,
                        help="specify the location for saving the model file")

    args = parser.parse_args()

    if not os.path.isfile(args.in_csv):
        raise Exception("Invalid input file (%s)" % args.in_csv)

    if not os.path.isdir(os.path.dirname(args.model_out)):
        raise Exception("model out directory does not exists: (%s)" % args.model_out)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(os.path.basename(__file__))

    with open(args.in_csv, "r") as fp:
        data = np.loadtxt(fp, delimiter=',')
    logger.info("data file loaded successfully (%dx%d)" % data.shape)

    assert (args.nepoch > 50) and (args.batch_size > 0)

    n, w = data.shape
    x = data[:, :-1].reshape((n, w-1))
    y = data[:, -1].reshape((n, 1))

    logger.info('starting the training process!!')

    if args.model_type == 'mlp':
        model = cg.MLP(epochs=args.nepoch, batch_size=args.batch_size)
        model.set_input_shape(w-1)
        model.set_output_shape(1)
        model.add_layers(10)
        model.add_layers(4)
        model.add_layers(4)
        model.set_model()
        model.summary()
    else:
        model = reg.XGBoost()
    acc = model.fit(x, y)
    logger.warning("Attained accuracy for the mode [%s] %.3f" % (args.model_out, acc))
    model.save(args.model_out)



