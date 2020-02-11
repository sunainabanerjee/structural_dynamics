import numpy as np
import pandas as pd


__version__ = "1.0"
__all__ = ['binary_roc', 'evaluate_roc', 'auc', 'bias_correction']


def binary_roc(y_true, y_proba, n=20):
    assert isinstance(y_true, (list, np.ndarray))
    assert isinstance(y_proba, (list, np.ndarray))
    assert len(y_true) == len(y_proba)
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    assert len(np.where((y_proba < 0) & (y_proba > 1))[0]) == 0
    assert (n > 0) and (n < len(y_true))
    assert len(np.unique(y_true)) == 2
    thetas = np.array(sorted(y_proba))[np.linspace(0, len(y_true)-1, n+1).astype(dtype=int)]

    classes = np.unique(y_true)
    actual = {tag: len(np.where(y_true == tag)[0]) for tag in classes}

    tpr, fpr = list(), list()
    for theta in thetas:
        p_idx = np.where(y_proba < theta)[0]
        if len(p_idx) == 0:
            tpr.append(0)
            fpr.append(0)
        if len(p_idx) > 0:
            predicted = {tag: len(np.where(y_true[p_idx] == tag)[0]) for tag in actual}
            tpr.append(predicted[classes[0]]/actual[classes[0]])
            fpr.append(predicted[classes[1]]/actual[classes[1]])
    return pd.DataFrame({'theta': thetas, 'tpr': tpr, 'fpr': fpr})


def evaluate_roc(y_true,
                 y_proba,
                 report_combined=True,
                 classes=None,
                 n=None):
    assert isinstance(y_proba, (np.ndarray, list))
    assert isinstance(y_true, (np.ndarray, list))
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    assert y_true.shape[0] == y_proba.shape[0]
    if n is None:
        n = 20
    if len(y_proba.shape) == 1:
        return binary_roc(y_true, y_proba, n)
    else:
        n_classes = y_proba.shape[1]
        assert len(np.unique(y_true)) == n_classes
        if classes is None:
            classes = np.unique(y_true)
        assert len(classes) == n_classes
        results = dict()
        for i, tag in enumerate(classes):
            y_true_mod = y_true.copy()
            y_true_mod[np.where(y_true != tag)[0]] = 0
            y_true_mod[np.where(y_true == tag)[0]] = 1
            results[tag] = binary_roc(y_true_mod, y_proba[:, i], n)
        if report_combined is True:
            tpr = np.linspace(0, 1, n+1)
            fpr = np.array([np.interp(tpr, results[cls]['tpr'].tolist(),
                                      results[cls]['fpr'].tolist()) for cls in results])
            fpr = np.mean(fpr, axis=0)
            return pd.DataFrame({'tpr': tpr.tolist(),
                                 'fpr': fpr.tolist()})
        else:
            return results


def auc(x, y):
    assert len(x) == len(y)
    x, y = np.array(x), np.array(y)
    x_idx = np.argsort(x)
    x = x[x_idx]
    y = y[x_idx]
    s = 0
    for i in range(1, len(x)):
        s += 0.5 * (x[i] - x[i-1]) * (y[i] + y[i-1])
    return s


def bias_correction(p_sample, p_prediction=None):
    assert isinstance(p_sample, np.ndarray) and len(p_sample.shape) == 2
    if p_prediction is None:
        p_prediction = p_sample.copy()

    assert isinstance(p_prediction, np.ndarray) and len(p_prediction.shape) == 2
    assert all(np.abs(np.sum(p_sample, axis=1) - 1) < 1e-3)
    assert all(np.abs(np.sum(p_prediction, axis=1) - 1) < 1e-3)
    assert p_sample.shape[1] == p_prediction.shape[1]

    bias = np.mean(p_sample, axis=0)
    balanced = np.repeat(1./p_sample.shape[1], p_sample.shape[1])
    return (p_prediction - bias) + balanced

