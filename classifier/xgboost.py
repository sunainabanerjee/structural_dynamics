import os
import logging
import pickle
import numpy as np
import xgboost as xgb
from .utility import *
from .classifier import Classifier
from sklearn.model_selection import train_test_split

__version__ = "1.0"
__all__ = ['XGBoost']


def class_prediction_accuracy(actual_tag, predicted_proba, classes, cutoff_proba=None):
    assert isinstance(actual_tag, np.ndarray)
    assert isinstance(predicted_proba, np.ndarray)
    assert actual_tag.shape[0] == predicted_proba.shape[0]
    if isinstance(classes, list):
        classes = np.array(classes)
    assert isinstance(classes, np.ndarray) and len(classes.shape) == 1
    assert classes.shape[0] == predicted_proba.shape[1]
    n, w = predicted_proba.shape
    if cutoff_proba is None:
        cutoff_proba = 1./w
    assert (cutoff_proba > 0) and (cutoff_proba < 1.)
    matched = 0
    for i in range(n):
        if actual_tag[i] in classes[predicted_proba[i] > cutoff_proba]:
            matched += 1
    return matched * (1.0/n)


class XGBoost(Classifier):
    def __init__(self,
                 n_class,
                 booster=XGBooster.gbtree(),
                 nthread=3,
                 max_depth=6,
                 learning_rate=0.2,
                 n_estimators=250,
                 subsample=0.8,
                 colsample_bytree=0.7,
                 objective=LossFunction.softprob()):
        assert booster in XGBooster.supported_list()
        assert objective in LossFunction.supported_list(filter='xgb')
        assert (nthread > 1) and (max_depth > 1)
        assert n_class >= 2
        assert learning_rate > 0
        assert (subsample > 0) and (subsample <= 1)
        assert (colsample_bytree > 0) and (colsample_bytree <= 1)
        if objective == LossFunction.binary():
            assert n_class == 2
        assert n_estimators > 10
        self.__name = 'xgb'
        self.__input_dim = None
        self.__output_dim = None
        self.__model = None
        self.__n_class = int(n_class)
        self.__booster = booster
        self.__thread = int(nthread)
        self.__lr = learning_rate
        self.__subsample = subsample
        self.__objective = objective
        self.__n_estimators = int(n_estimators)
        self.__max_depth = int(max_depth)
        self.__colsample_bytree = colsample_bytree
        self.__logger = logging.getLogger(os.path.basename(__file__)[:-3])
        self.__pthresh = 1.0 / self.__n_class

    @property
    def pthresh(self):
        return self.__pthresh

    def set_pthresh(self, pthresh):
        assert (pthresh >= 0) and (pthresh <= 1.0)
        self.__pthresh = pthresh

    @property
    def name(self):
        return self.__name

    @property
    def input_shape(self):
        return self.__input_dim

    @property
    def out_classes(self):
        return self.__output_dim

    @property
    def out_class_names(self):
        assert self.__model is not None
        return self.__model.classes_

    @property
    def is_set(self):
        return self.__model is not None

    def fit(self, x, y, test_split=0.2):
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
        assert (test_split > 0) and (test_split <= 0.5)
        assert (len(x.shape) == 2) and (len(y.shape) == 1)
        assert (x.shape[0] == y.shape[0])
        if len(np.unique(y)) != self.__n_class:
            raise Exception("Improper class definition")

        self.__model = xgb.XGBClassifier(booster=self.__booster,
                                         learning_rate=self.__lr,
                                         max_depth=self.__max_depth,
                                         n_estimators=self.__n_estimators,
                                         objective=self.__objective,
                                         subsample=self.__subsample,
                                         colsample_bytree=self.__colsample_bytree,
                                         nclass=self.__n_class,
                                         nthread=self.__thread)
        self.__input_dim = x.shape[1]
        self.__output_dim = y.shape[0]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split)
        self.__model.fit(x_train, y_train)
        accuracy = class_prediction_accuracy(y_test,
                                             self.__model.predict_proba(x_test),
                                             self.__model.classes_,
                                             self.__pthresh)
        np.sum(y_test == self.predict(x_test))/len(y_test)
        self.__logger.debug("Accuracy: %.2f" % accuracy)
        return accuracy

    def predict(self, x):
        assert self.is_set
        assert isinstance(x, np.ndarray)
        assert (len(x.shape) == 2)
        return self.__model.predict(x)

    def predict_proba(self, x):
        assert self.is_set
        assert isinstance(x, np.ndarray)
        assert len(x.shape) == 2
        return self.__model.predict_proba(x)

    def save(self, file_path):
        assert self.is_set
        assert os.path.isdir(os.path.dirname(os.path.abspath(file_path)))
        if os.path.isfile(file_path):
            self.__logger.warning("Over writing model file [%s]" % file_path)
        pickle.dump(self.__model, open(file_path, "wb"))

    def load(self, model_path):
        assert os.path.isfile(model_path)
        self.__model = pickle.load(open(model_path, "rb"))

    @property
    def loading_factors(self):
        assert self.__model is not None
        return self.__model.coef_

    def get_xgb_model(self):
        return self.__model




