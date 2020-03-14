import os
import numpy as np
import pandas as pd
import seaborn as sns
import regressor as reg
from matplotlib import pyplot as plt


if __name__ == "__main__":
    mol_type = 'atp'
    data_file = os.path.join(os.path.dirname(__file__),
                             '..',
                             'scripts',
                             'site_signature',
                             '%s_regression_feature_vector_with_tag_mar2020.csv' % mol_type)
    assert os.path.isfile(data_file)
    data = pd.read_csv(data_file, header=None, sep=',')
    mol_type, max_dev = 'atp', 10
    data_tag = data.iloc[:, 0].tolist()
    mutants = np.unique([dt.split("_")[0] for dt in data_tag]).tolist()

    y_preds = []

    for mutant in mutants:
        data_prune = data[data[0].str.contains(mutant)]
        data_prune = data_prune.iloc[:, 1:].values
        y_expected = np.unique(data_prune[:, -1])[0]
        x = data_prune[:, :-1]

        model_in = os.path.join(os.path.dirname(__file__),
                                'out',
                                'revised_%s_%d_regression.dat' % (mol_type, max_dev))

        assert os.path.isfile(model_in)
        model = reg.XGBoost()
        model.load(model_in)

        y_estimated = model.predict(x)
        y_estimated = np.clip(y_estimated,
                              a_min=-0.9999,
                              a_max=0.9999)
        y_estimated = np.arctanh(y_estimated) * 200 + 100
        y_expected = np.arctanh(y_expected)*200 + 100
        y_predicted = np.mean(y_estimated)
        y_lower, y_upper = np.quantile(y_estimated, q=(0.1, 0.9) )
        y_preds.append((mutant, y_expected, y_predicted, y_lower, y_upper))

    plt.plot([y[1] for y in y_preds], [y[2] for y in y_preds], 'or')
    for i, y in enumerate(y_preds):
        plt.vlines(x=y[1], ymin=y[3], ymax=y[4])
    plt.grid("on")
    plt.xlabel('Experimental')
    plt.ylabel('Predicted')
    plt.show()

    df = pd.DataFrame({'mutant': [y[0] for y in y_preds],
                       'experimental': [y[1] for y in y_preds],
                       'predicted': [y[2] for y in y_preds],
                       'lower': [y[3] for y in y_preds],
                       'upper': [y[4] for y in y_preds]})

    prediction_out = os.path.join(os.path.dirname(__file__),
                                  'out',
                                  '%s_model_%.2f_prediction.csv' % (mol_type, max_dev))
    df.to_csv(prediction_out, sep=',', header=True, index=False)