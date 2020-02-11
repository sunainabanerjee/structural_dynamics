import os
import shap
import numpy as np
import pandas as pd
from classifier import XGBoost, auc
from scipy.cluster import hierarchy
from matplotlib import pyplot as plt
from scipy.spatial.distance import  pdist
from mutant_model import BulkSignatureGenerator


def cluster_information(z, breaks=10):
    assert isinstance(z, np.ndarray) and (len(z.shape) == 2) and (z.shape[1] == 4)
    n = z.shape[0]
    h_min, h_max = 0, z[-1, 2]
    x, y = list(), list()
    for h in np.linspace(h_min, h_max, breaks + 1):
        x.append((h - h_min)/(h_max - h_min))
        y.append(len(np.where(z[:, 2] >= h)[0])/n)
    return auc(x, y)


if __name__ == "__main__":
    mol_type = 'rna'
    model_file = os.path.join(os.path.dirname(__file__),
                              'out',
                              'revised_%s_classify.dat' % mol_type)

    assert os.path.isfile(model_file)
    data_file = os.path.join(os.path.dirname(__file__),
                             '..',
                             'scripts',
                             'site_signature',
                             '%s_feature_vector_with_tag_jan2020.csv' % mol_type)

    assert os.path.isfile(data_file)
    data = pd.read_csv(data_file, header=None, sep=',')
    tags = [fld.split("_")[0] for fld in data[0].tolist()]
    n, f = data.shape
    indices = list(data.keys())[1:]
    data = data[indices].values
    n, f = data.shape
    x = data[:, :-1].reshape((n, f - 1))
    y = data[:, -1].tolist()

    cls = XGBoost(n_class=3)
    cls.load(model_file)

    model = cls.get_xgb_model()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x, approximate=True)
    proba = cls.predict_proba(x)
    n_data, n_class = proba.shape

    global_shap = np.multiply(proba[:, 0].reshape((n_data, 1)), shap_values[0]) + \
                  np.multiply(proba[:, 1].reshape((n_data, 1)), shap_values[1]) + \
                  np.multiply(proba[:, 2].reshape((n_data, 1)), shap_values[2])

    dim_names = BulkSignatureGenerator.dim_names()

    cor = np.array([np.corrcoef(global_shap[:, i], x[:, i])[0, 1] for i in range(len(dim_names))])
    cor[np.isnan(cor)] = 0

    plt.hist(cor, bins=25)
    plt.title("Factor to feature value correlation distribution")
    plt.show()

    top_k_factors = len(np.where(np.abs(cor) >= 0.35)[0])
    print(top_k_factors)

    important_factors = np.argsort(np.abs(cor))[-top_k_factors:]
    global_shap = global_shap[:, important_factors]
    important_dims = [dim_names[i] for i in important_factors]

    x_df = pd.DataFrame(x[:, important_factors], columns=important_dims)
    shap.summary_plot(global_shap,
                      x_df,
                      max_display=10,
                      plot_type='violin',
                      class_names=['deficient', 'basal', 'enhanced'])
    sorted_index = np.argsort(np.abs(global_shap), axis=1)

    def set_dist(x1, x2):
        assert len(x1) == len(x2)
        return len(set(x1).difference(x2))

    best_k, best_score = None, None
    best_distance_score = None
    for top_k_set in range(int(top_k_factors*0.25), int(top_k_factors*0.75)):
        screened_index = sorted_index[:, -top_k_set:]
        all_dist = pdist(screened_index[:n_data], lambda u, v: set_dist(u, v))
        z = hierarchy.linkage(all_dist,
                              method='complete')
        score = cluster_information(z, breaks=top_k_factors)
        print(top_k_set, score)
        if (best_score is None) or (best_score > score):
            best_score, best_k = score, top_k_set
            best_distance_score = all_dist.copy()
    print(best_k)
    z = hierarchy.linkage(best_distance_score, method='complete')
    hierarchy.dendrogram(z)
    plt.show()


