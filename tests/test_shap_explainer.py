import os
import shap
import numpy as np
import pandas as pd
from classifier import XGBoost
from mutant_model import BulkSignatureGenerator


if __name__ == "__main__":
    mol_type = 'atp'
    top_display = 10
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
    shap_values = explainer.shap_values(x_test, approximate=True)
    x_test = pd.DataFrame(x_test, columns=dim_names)
    if plot_type == 'bar':
        shap.summary_plot(shap_values,
                          x_test,
                          max_display=top_display,
                          plot_type='bar',
                          class_names=['deficient', 'basal', 'enhanced'],
                          color=lambda i: list(["#b09c8599","#dc000099","#8491b4ff",])[i])
    else:
        for i in range(len(shap_values)):
            shap.summary_plot(shap_values[i],
                              x_test,
                              max_display=top_display,
                              plot_type='violin',
                              class_names=['deficient', 'basal', 'enhanced'])




