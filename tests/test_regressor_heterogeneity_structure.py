import os
import numpy as np
import pandas as pd
import regressor as reg
from collections import Counter
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

    def check_snapshot(x):
        if "cryst" in x:
            return True
        start, end = float(x.split("_")[2][1:]) , float(x.split("_")[3])
        return (end - start) <= 1

    def time_tag(x):
        if "cryst" in x:
            return 0
        tag = x.split("_")[2][0]
        start = 0 if tag == 'e' else 4
        return start + 0.5*(float(x.split("_")[2][1:]) + float(x.split("_")[3]))


    valid_rows = pd.Series([check_snapshot(x) for x in data_tag])
    data = data[valid_rows]

    data_tag = data.iloc[:, 0]
    y_expected = np.arctanh(data.iloc[:, -1].values)*200 + 100
    x = data.iloc[:, 1:-1].values
    y_time = [time_tag(t) for t in data_tag]

    model_in = os.path.join(os.path.dirname(__file__),
                            'out',
                            'revised_%s_%d_regression.dat' % (mol_type, max_dev))
    assert os.path.isfile(model_in)
    model = reg.XGBoost()
    model.load(model_in)
    eps = 1e-6
    y_estimated = np.arctanh(np.clip(model.predict(x), a_min=-1. + eps, a_max=1. - eps))*200 + 100

    y_err = np.abs(y_expected - y_estimated)
    plt.plot(y_time, y_err, '.r')
    plt.xticks(np.arange(0, max(y_time), 0.5))
    plt.grid("on")
    plt.show()


