import os
import logging
import numpy as np
import pickle
import xgboost as xgb
from .regressor import Regressor
from .utility import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import  MultiOutputRegressor

__version__ = "1.0"
__all__ = ['XGBoost']


class XGBoost(Regressor):
    def __init__(self,
                 booster='dart',
                 nthread=3,
                 max_depth=6,
                 learning_rate=0.2,
                 n_estimators=250,
                 subsample=0.8,
                 colsample_bytree=0.7,
                 objective=LossFunction.squared()):
        assert booster in XGBooster.supported_list()
        assert objective in LossFunction.supported_list(filter='xgb')
        assert (nthread > 1) and (max_depth > 1)
        assert learning_rate > 0
        assert (subsample > 0) and (subsample <= 1)
        assert (colsample_bytree > 0) and (colsample_bytree <= 1)
        assert n_estimators >= 50
        self.__name = 'xgb'
        self.__input_dim = None
        self.__output_dim = None
        self.__model = None
        self.__booster = booster
        self.__thread = int(nthread)
        self.__objective = objective
        self.__lr = learning_rate
        self.__subsample = subsample
        self.__nestimators = int(n_estimators)
        self.__max_depth = int(max_depth)
        self.__colsample_bytree = colsample_bytree
        self.__logger = logging.getLogger(os.path.basename(__file__)[:-3])

    @property
    def name(self):
        return self.__name

    @property
    def input_shape(self):
        return self.__input_dim

    @property
    def output_shape(self):
        return self.__output_dim

    @property
    def is_set(self):
        return self.__model is not None

    def fit(self, x, y, test_split=0.2):
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
        assert (test_split > 0) and (test_split <= 0.5)
        assert (len(x.shape) == 2) and (len(y.shape) == 2)
        assert (x.shape[0] == y.shape[0]) and (y.shape[1] > 0)
        if y.shape[1] == 1:
            self.__model = xgb.XGBRegressor(booster=self.__booster,
                                            learning_rate=self.__lr,
                                            max_depth=self.__max_depth,
                                            n_estimators=self.__nestimators,
                                            subsample=self.__subsample,
                                            objective=self.__objective,
                                            colsample_bytree=self.__colsample_bytree,
                                            nthread=self.__thread)
        else:
            self.__model = MultiOutputRegressor(xgb.XGBRegressor(booster=self.__booster,
                                                                 learning_rate=self.__lr,
                                                                 max_depth=self.__max_depth,
                                                                 n_estimators=self.__nestimators,
                                                                 subsample=self.__subsample,
                                                                 objective=self.__objective,
                                                                 colsample_bytree=self.__colsample_bytree,
                                                                 nthread=self.__thread))
        self.__input_dim = x.shape[1]
        self.__output_dim = y.shape[1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split)
        self.__model.fit(x_train, y_train)
        accuracy = mean_absolute_error(y_test, self.predict(x_test))
        self.__logger.debug("Prediction Error: %.2f" % accuracy)
        return accuracy

    def predict(self, x):
        assert self.is_set
        assert isinstance(x, np.ndarray)
        assert (len(x.shape) == 2)
        res = self.__model.predict(x)
        return res

    def save(self, file_path):
        assert self.is_set
        assert os.path.isdir(os.path.dirname(os.path.abspath(file_path)))
        if os.path.isfile(file_path):
            self.__logger.warning("Over writing model file [%s]" % file_path)
        pickle.dump(self.__model, open(file_path, "wb"))

    def load(self, model_path):
        assert os.path.isfile(model_path)
        self.__model = pickle.load(open(model_path, "rb"))



