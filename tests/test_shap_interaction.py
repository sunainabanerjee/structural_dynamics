import os
import shap
import numpy as np
import pandas as pd
from classifier import XGBoost
from mutant_model import BulkSignatureGenerator


if __name__ == "__main__":
    mol_type = 'atp'
    model_file = os.path.join(os.path.dirname(__file__),
                              'out',
                              'revised_%s_classify.dat' % mol_type)
    assert os.path.isfile(model_file)
    data_file = os.path.join(os.path.dirname(__file__),
                             '..',
                             'scripts',
                             'site_signature',
                             'revised_%s_site_signature_without_tag.csv' % mol_type)
    assert os.path.isfile(data_file)
    data = np.loadtxt(data_file, delimiter=',')
    np.random.shuffle(data)
    n, f = data.shape
    x = data[:, :-1].reshape((n, f - 1))
    y = data[:, -1]

    plot_type = 'bar'
    cls = XGBoost(n_class=3)
    cls.load(model_file)
    model = cls.get_xgb_model()
    explainer = shap.TreeExplainer(model)
    test = np.random.choice(range(x.shape[0]), 1000)
    x_test, y_test = x[test], y[test]
    dim_names = BulkSignatureGenerator.dim_names()
    shap_int_values = explainer.shap_interaction_values(x_test)
    print(shap_int_values)

