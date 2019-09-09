from abc import ABC
from abc import abstractmethod

__version__ = "1.0"
__all__ = ['Regressor']


class Regressor(ABC):
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def fit(self, x, y, test_split=0.2):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def save(self, file_path):
        pass

    @abstractmethod
    def load(self, model_path):
        pass

