import os
import shap
import numpy as np
import pandas as pd
from regressor import XGBoost
from mutant_model import BulkSignatureGenerator
from matplotlib import pyplot as plt


if __name__ == "__main__":
    mol_type = 'atp'
    top_display = 30
    max_dev = 10
    model_file = os.path.join(os.path.dirname(__file__),
                              'out',
                              'revised_%s_%d_regression.dat' % (mol_type, max_dev))
    assert os.path.isfile(model_file)
    data_file = os.path.join(os.path.dirname(__file__),
                             '..',
                             'scripts',
                             'site_signature',
                             '%s_regression_feature_vector_without_tag_mar2020.csv' % mol_type)
    assert os.path.isfile(data_file)
    data = np.loadtxt(data_file, delimiter=',')
    np.random.shuffle(data)
    n, f = data.shape
    x = data[:, :-1].reshape((n, f - 1))
    y = data[:, -1]

    cls = XGBoost()
    cls.load(model_file)
    model = cls.get_xgb_model()
    explainer = shap.TreeExplainer(model)
    test = np.random.choice(range(x.shape[0]), 1000)
    x_test, y_test = x[test], y[test]
    dim_names = BulkSignatureGenerator.dim_names()
    shap_values = explainer.shap_values(x_test, approximate=True)
    x_test = pd.DataFrame(x_test, columns=dim_names)

    print(dim_names)
    '''
    matplotlib.use('TkAgg')
    shap.initjs()
    plt.interactive(False)
    shap.force_plot(explainer.expected_value, shap_values, x_test)
    plt.show()
    '''

    shap.summary_plot(shap_values,
                      x_test,
                      max_display=top_display,
                      plot_type='violin',
                      class_names=['deficient', 'basal', 'enhanced'])




