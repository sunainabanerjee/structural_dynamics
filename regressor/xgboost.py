import os
import logging
import numpy as np
import xgboost as xgb
from .regressor import Regressor
from .utility import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

__version__ = "1.0"
__all__ = ['XGBoost']


class XGBoost(Regressor):
    def __init__(self,
                 booster='dart',
                 nthread=2,
                 max_depth=6,
                 objective=LossFunction.squared()):
        assert booster in XGBooster.supported_list()
        assert objective in LossFunction.supported_list(filter='xgb')
        assert (nthread > 1) and (max_depth > 1)
        self.__name = 'xgb'
        self.__input_dim = None
        self.__output_dim = None
        self.__model = None
        self.__booster = booster
        self.__thread = int(nthread)
        self.__objective = objective
        self.__max_depth = int(max_depth)
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
        assert x.shape[0] == y.shape[0]
        self.__model = xgb.XGBRegressor(booster=self.__booster,
                                        max_depth=self.__max_depth,
                                        objective=self.__objective)
        self.__input_dim = x.shape[1]
        self.__output_dim = y.shape[1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split)
        self.__model.fit(x_train, y_train)
        print(x_test.shape)
        print(len(self.predict(x_test)))
        accuracy = mean_absolute_error(y_test, self.predict(x_test))
        self.__logger.debug("Accuracy: %.2f" % accuracy)
        return accuracy

    def predict(self, x):
        assert self.is_set
        assert isinstance(x, np.ndarray)
        assert (len(x.shape) == 2)
        return self.__model.predict(x)

    def save(self, file_path):
        assert self.is_set
        assert os.path.isdir(os.path.dirname(os.path.abspath(file_path)))
        if os.path.isfile(file_path):
            self.__logger.warn("Over writing model file [%s]" % file_path)
        self.__model.save_model(file_path)

    def load(self, model_path):
        assert os.path.isfile(model_path)
        self.__model = xgb.XGBRegressor(booster=self.__booster,
                                        nthread=self.__thread,
                                        max_depth=self.__max_depth)
        self.__model.load_model(model_path)



