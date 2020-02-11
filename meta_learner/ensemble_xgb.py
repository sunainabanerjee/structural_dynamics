import re
import os
import pickle
import logging
from classifier import XGBoost, XGBooster
from .balanced_sampler import BalancedClassSampler

__version__ = "1.0"
__all__ = ['MetaLearnerXGBoost']


class MetaLearnerXGBoost:
    def __init__(self, n_ensemble=100):
        n_ensemble = int(n_ensemble)
        assert n_ensemble > 1
        self.__sampler = None
        self.__ensemble_size = n_ensemble
        self.__ensemble = []
        self.__factor = 3.0
        self.__logger = logging.getLogger(self.__class__.__name__)

    @property
    def size(self):
        return len(self.__ensemble)

    @property
    def out_classes(self):
        assert self.size > 0
        return self.__ensemble[0].out_classes

    @property
    def out_class_names(self):
        assert self.size > 0
        return self.__ensemble[0].out_class_names

    def ensemble_size(self, n=None):
        if n is None:
            return self.__ensemble_size
        else:
            n = int(n)
            assert n > 1
            self.__ensemble_size = n

    def reset(self):
        self.__sampler = None
        self.__ensemble.clear()

    def minimum_datasize_factor(self, f=None):
        if f is None:
            return self.__factor
        else:
            assert f > 0
            self.__factor = float(f)

    def fit(self, x, y):
        if self.size > 0:
            raise SystemError("System is already trained, please call reset function to erase the memory explicitly")
        self.__sampler = BalancedClassSampler(x, y)
        n, dim = x.shape
        assert n > self.__factor * dim
        sz = int(self.__factor * dim)
        self.__logger.debug("Iterating with sample set size: (%d)" % sz)
        print("Training set size: [%d]" % sz)
        for i in range(self.__ensemble_size):
            self.__logger.debug("Creating ensemble model - %d" % i )
            model = XGBoost(n_class=self.__sampler.n_classes,
                            booster=XGBooster.gbtree(),
                            n_estimators=100)
            sx, sy = self.__sampler.sample(sz)
            model.fit(sx, sy)
            self.__ensemble.append(model)
            self.__logger.debug("Model-%d completed!" % i)

    def sample(self, n):
        assert self.__sampler is not None
        return self.__sampler.sample(n)

    def __getitem__(self, item):
        n = int(item)
        assert (n >= 0) and (n < self.size)
        return self.__ensemble[n]

    def save(self, out_dir):
        if self.size == 0:
            raise SystemError("Model is not trained on data!")
        assert os.path.isdir(out_dir)
        model_name = "xgb_model_{}.dat"
        for i, model in enumerate(self.__ensemble):
            model_file = os.path.join(out_dir, model_name.format(i))
            model.save(model_file)
        sampler_out = "xgb_sampler.pkl"
        with open(os.path.join(out_dir, sampler_out), 'wb') as f:
            pickle.dump(self.__sampler, f)

    def load(self, in_dir):
        assert os.path.isdir(in_dir)
        sample_file = "xgb_sampler.pkl"
        assert os.path.isfile(os.path.join(in_dir, sample_file))
        with open(os.path.join(in_dir, sample_file), "rb") as f:
            self.__sampler = pickle.load(f)
        model_name_pattern = "^xgb_model_\\d+.dat$"
        model_files = [filename for filename in os.listdir(in_dir) if re.match(model_name_pattern, filename)]
        assert len(model_files) > 0
        self.__ensemble = list()
        for f in model_files:
            model = XGBoost(n_class=self.__sampler.n_classes,
                            booster=XGBooster.gbtree(),
                            n_estimators=100)
            model.load(os.path.join(in_dir, f))
            self.__ensemble.append(model)


