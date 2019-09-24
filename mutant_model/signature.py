import numpy as np
import geometry as geo
from copy import deepcopy
from .system_constants import *
from geometry import Coordinate3d


__version__ = "1.0"
__all__ = ['Signature', 'is_compatible', 'signature_diff', 'merge_signature']


class Signature:
    def __init__(self, min_coord, max_coord, size=None):
        assert hasattr(min_coord, '__len__') and len(min_coord) == 3
        assert hasattr(max_coord, '__len__') and len(max_coord) == 3
        if size is None:
            size = GridPropertyLookup.signature_grid_dim()
        if not hasattr(size, '__len__'):
            size = (size, size, size)
        assert all([min_coord[i] < max_coord[i] for i in range(3)])
        self.__min_coordinate = Coordinate3d(*min_coord)
        self.__max_coordinate = Coordinate3d(*max_coord)
        self.__gdim = size
        assert all([s > 1 for s in self.__gdim])
        dx = (self.__max_coordinate[0] - self.__min_coordinate[0]) / self.__gdim[0]
        dy = (self.__max_coordinate[1] - self.__min_coordinate[1]) / self.__gdim[1]
        dz = (self.__max_coordinate[2] - self.__min_coordinate[2]) / self.__gdim[2]
        self.__gsize = (dx, dy, dz)
        self.__signature = {}

    @property
    def grid_size(self):
        return self.__gsize

    @property
    def grid_dim(self):
        return self.__gdim

    def inside(self, x, y, z):
        crd = [x, y, z]
        bounds = [((crd[i] > self.__min_coordinate[i]) and
                   (crd[i] < self.__max_coordinate[i])) for i in range(len(crd))]
        return all(bounds)

    def __lidx(self, ix, iy, iz):
        assert (ix < (self.__gdim[0] - 1)) and (ix >= 0)
        assert (iy < (self.__gdim[1] - 1)) and (iy >= 0)
        assert (iz < (self.__gdim[2] - 1)) and (iz >= 0)
        return ix * (self.__gdim[1] - 1) * (self.__gdim[2] - 1) + iy * (self.__gdim[2] - 1) + iz

    def in_index(self, x, y, z):
        possible_indices = []
        if self.inside(x, y, z):
            ix = int((x - self.__min_coordinate.x) // self.__gsize[0])
            iy = int((y - self.__min_coordinate.y) // self.__gsize[1])
            iz = int((z - self.__min_coordinate.z) // self.__gsize[2])
            x_set = [ix] if ix == 0 else [ix - 1] if ix == (self.__gdim[0]-1) else [ix, ix - 1]
            y_set = [iy] if iy == 0 else [iy - 1] if iy == (self.__gdim[1]-1) else [iy, iy - 1]
            z_set = [iz] if iz == 0 else [iz - 1] if iz == (self.__gdim[2]-1) else [iz, iz - 1]
            for i in x_set:
                for j in y_set:
                    for k in z_set:
                        possible_indices.append(self.__lidx(i, j, k))
        return possible_indices

    def add(self, x, y, z, v):
        indices = self.in_index(x, y, z)
        for idx in indices:
            if idx not in self.__signature:
                self.__signature[idx] = []
            self.__signature[idx].append(v)

    @property
    def size(self):
        return (self.__gdim[0] - 1) * (self.__gdim[1] - 1) * (self.__gdim[2] - 1)

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        if (item < 0) or (item >= self.size):
            raise IndexError("Index out of bound!")
        return self.__signature[item] if item in self.__signature else []

    def append2list_(self, item, data, operation):
        assert operation in {"concat", "max", "min", "add"}
        if hasattr(data, '__len__'):
            data = list(data)
        else:
            data = [data]
        assert isinstance(data, list)
        if (item < 0) or (item >= self.size):
            raise IndexError("Index out of bound!")
        if operation == "concat":
            if item not in self.__signature:
                self.__signature[item] = []
            self.__signature[item] = self.__signature[item] + data
        else:
            assert item in self.__signature
            assert len(self.__signature[item]) % len(data) == 0
            if len(data) != len(self.__signature[item]):
                n = int(len(self.__signature[item]) // len(data))
                data = data * n
            if operation == "max":
                self.__signature[item] = np.maximum(self.__signature[item], data)
            elif operation == "min":
                self.__signature[item] = np.minimum(self.__signature[item], data)
            elif operation == "add":
                self.__signature[item] = np.array(self.__signature[item]) + np.array(data)

    @property
    def min_coordinate(self):
        return self.__min_coordinate

    @property
    def max_coordinate(self):
        return self.__max_coordinate

    def tolist(self):
        return [(i, np.max(self.__signature[i])) for i in range(self.size)]


def is_compatible(signature1, signature2):
    assert isinstance(signature1, Signature)
    assert isinstance(signature2, Signature)
    if geo.distance(signature1.min_coordinate, signature2.min_coordinate) < 1e-3:
        if geo.distance(signature1.max_coordinate, signature2.max_coordinate) < 1e-3:
            if len(signature1) == len(signature2):
                return all([(len(signature1[i]) == len(signature2[i]))
                            for i in range(len(signature1))])
    return False


def merge_signature(signature1, signature2, operation="concat"):
    assert operation in {"concat", "max", "min", "add"}
    assert isinstance(signature1, Signature)
    assert isinstance(signature2, Signature)
    assert geo.distance(signature1.min_coordinate, signature2.min_coordinate) < 1e-3
    assert geo.distance(signature1.max_coordinate, signature2.max_coordinate) < 1e-3
    assert len(signature1) == len(signature2)
    msig = deepcopy(signature1)
    for i in range(len(signature2)):
        msig.append2list_(i, signature2[i], operation=operation)
    return msig


def signature_diff(signature1, signature2):
    assert isinstance(signature1, Signature)
    assert isinstance(signature2, Signature)
    assert len(signature1) == len(signature2)
    score = []
    for i in range(len(signature1)):
        x = np.array(signature1[i], dtype=np.float) - np.array(signature2[i], dtype=np.float)
        score.append((np.sqrt(np.mean(np.square(x))), np.std(x)))
    return score

