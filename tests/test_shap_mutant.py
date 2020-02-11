import os
import shap
import numpy as np
import pandas as pd
from classifier import XGBoost
from matplotlib import pyplot as plt
from mutant_model import BulkSignatureGenerator


if __name__ == "__main__":
    mol_type = 'atp'
    mutant = 'W501F'
    top_display = 20
    model_file = os.path.join(os.path.dirname(__file__),
                              'out',
                              'revised_%s_classify.dat' % mol_type)
    assert os.path.isfile(model_file)
    data_file = os.path.join(os.path.dirname(__file__),
                             '..',
                             'scripts',
                             'site_signature',
                             '%s_site_signature_with_tag.csv' % mol_type)
    assert os.path.isfile(data_file)
    with open(data_file, "r") as fp:
        data = pd.read_csv(data_file, header=None)
    n, f = data.shape
    cnames = list(data.columns)
    x = data[cnames[1:-1]].values
    y = data[cnames[-1]].values.reshape(n)
    mutants = np.array([cname.split("_")[0] for cname in data[cnames[0]]])

    plot_type = 'violin'
    cls = XGBoost(n_class=3)
    cls.load(model_file)
    model = cls.get_xgb_model()
    class_order = model.classes_
    explainer = shap.TreeExplainer(model)
    test = np.where(mutants == mutant)[0]
    mutant_tag = np.unique(y[test])
    assert len(mutant_tag) == 1
    mutant_tag = mutant_tag[0]
    assert len(test) > 1
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
        for i, data_in in enumerate(shap_values):
            if class_order[i] == mutant_tag:
                plt.subplots_adjust(left=0.35, right=0.98)
                plt.title("Mutant: %s (%d)" % (mutant, mutant_tag))
                shap.summary_plot(data_in,
                                  x_test,
                                  max_display=top_display,
                                  plot_type='violin',
                                  class_names=['deficient', 'basal', 'enhanced'])




