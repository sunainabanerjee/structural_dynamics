import os
import json
import logging
import argparse
import numpy as np
import regressor as reg
import keras.backend as K


def absolute_sum_error(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to train the sasa models from feature csv")

    parser.add_argument('--in', action='store', dest='in_csv',
                        help="input csv, containing the feature description and regressor value",
                        type=str, required=True)

    parser.add_argument('--nepoch', action='store', dest='nepoch',
                        default=750,
                        help='number of epochs in training (default: 750)',
                        type=int, required=False)

    parser.add_argument('--batch-size', action='store', dest='batch_size',
                        default=750,
                        help="number of data each batch (default: 750)",
                        type=int, required=False)

    parser.add_argument('--model-out', action='store', dest='model_out',
                        type=str, required=True,
                        help="specify the location for saving the model file")

    parser.add_argument('--n-features', action='store', dest='nfeatures',
                        type=int, required=False, default=60,
                        help="Number of columns to be used as features (default: 60)")

    parser.add_argument('--train-history', action='store', dest='history_file',
                        type=str, required=False,
                        help="If provided model training history per epoch will be stored as json")

    parser.add_argument('--seed', action='store', dest='seed',
                        type=int, required=False, default=73,
                        help="Random number seed (default: 73)")

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

    f = args.nfeatures
    n, w = data.shape
    assert w > f
    rg_dim = w - f
    x = data[:, :f].reshape((n, f))
    y = data[:, f:].reshape((n, rg_dim))

    logger.info('starting the training process!!')
    np.random.seed(args.seed)

    model = reg.MLP(epochs=args.nepoch,
                    batch_size=args.batch_size)
    model.set_learning_rate(0.0001)
    model.set_input_shape(f)
    model.set_output_shape(rg_dim, activation=reg.Activations.relu())
    model.add_layers(12, layer_type=reg.MLPLayerType.dense(), activation=reg.Activations.relu())
    model.add_layers(8, layer_type=reg.MLPLayerType.dense(), activation=reg.Activations.relu())
    model.add_layers(4, layer_type=reg.MLPLayerType.dense(), activation=reg.Activations.relu())
    model.set_model()
    acc = model.fit(x, y)
    model.save(args.model_out)
    if (args.history_file is not None) and (os.path.isdir(os.path.dirname(args.history_file))):
        history = model.train_history
        with open(args.history_file, 'w') as fp:
            json.dump(history, fp, indent=2)



