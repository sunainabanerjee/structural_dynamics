import numpy as np

__version__ = "1.0"
__all__ = ['BalancedClassSampler']


class BalancedClassSampler:
    def __init__(self, x, y):
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
        assert len(y.shape) == 1
        y_u = np.unique(y)
        assert len(y_u) < len(y)
        self.__x = x.copy()
        self.__y = y.copy()
        self.__cls_lookup = dict()
        for cls in y_u:
            self.__cls_lookup[cls] = np.where(self.__y == cls)[0]

    @property
    def n_classes(self):
        return len(self.__cls_lookup)

    @property
    def data_size(self):
        return len(self.__x)

    @property
    def class_sizes(self):
        return {cls: len(self.__cls_lookup[cls]) for cls in self.__cls_lookup}

    @property
    def maximum_sample_size(self):
        return self.n_classes * np.min(self.class_sizes.values())

    def sample(self, n):
        n = int(n)
        assert n > len(self.__cls_lookup)
        classes = list(self.__cls_lookup.keys())
        sizes = [n // self.n_classes for cls in classes]
        counter = 0
        while np.sum(sizes) != n:
            sizes[counter] = sizes[counter] + 1
            counter = (counter + 1) % self.n_classes
        replace = [sz > len(self.__cls_lookup[classes[i]]) for i, sz in enumerate(sizes)]
        index = []
        for j, sz in enumerate(sizes):
            cls = classes[j]
            index = index + np.random.choice(self.__cls_lookup[cls], sz, replace=replace[j]).tolist()
        np.random.shuffle(index)
        return self.__x[index], self.__y[index]
