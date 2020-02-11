import re
import shap
import numpy as np
from matplotlib import pylab as plt
from .ensemble_xgb import MetaLearnerXGBoost

__version__ = "1.0"
__all__ = ['ensemble_analyzer']


def ensemble_analyzer(meta_learner, sample_size, method="rmse"):
    assert isinstance(meta_learner, MetaLearnerXGBoost)
    assert (method in ['rmse', 'supnorm']) or re.match('^setdiff_\\d+$', method)
    assert meta_learner.size == meta_learner.ensemble_size()
    x, y = meta_learner.sample(sample_size)
    shap_values = []
    for i in range(meta_learner.size):
        model = meta_learner[i].get_xgb_model()
        explainer = shap.TreeExplainer(model)
        shap_value = explainer.shap_values(x)
        shap_values.append(np.array(shap_value))
    distances = np.zeros((meta_learner.size, meta_learner.size), dtype=np.float)
    for i in range(meta_learner.size):
        feature_id = 10
        plt.scatter(x[:, feature_id], shap_values[i][0, :, feature_id], c='red', s=0.2)
        plt.scatter(x[:, feature_id], shap_values[i][1, :, feature_id], c='green', s=0.2)
        plt.scatter(x[:, feature_id], shap_values[i][2, :, feature_id], c='blue', s=0.2)
        plt.show()
        for j in range(i+1, meta_learner.size):
            if method == 'rmse':
                d = np.sqrt(np.mean(np.square(shap_values[i] - shap_values[j])))
            elif method == 'supnorm':
                d = np.max(np.mean(np.abs(shap_values[i] - shap_values[j]), axis=0))
            elif method.startswith('setdiff'):
                f = float(method.split("_")[1])/100
                n_classes = shap_values[i].shape[0]
                assert n_classes == shap_values[j].shape[0]
                dist = list()
                for k in range(n_classes):
                    n = int(len(shap_values[i][k]) * f)
                    list_i = set(np.argsort(np.abs(np.mean(shap_values[i][k], axis=0)))[::-1][:n])
                    list_j = set(np.argsort(np.abs(np.mean(shap_values[j][k], axis=0)))[::-1][:n])
                    dist.append( len(list_i.symmetric_difference(list_j)) / n)
                d = np.max(dist)
            distances[i, j] = d
            distances[j, i] = d
    return distances
