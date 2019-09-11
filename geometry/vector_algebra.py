import numpy as np
from .coordinate import *

__version__ = "1.0"
__all__ = ['Vector3d', 'dotp', 'crossp',
           'connecting_vector', 'point_vector',
           'plane_vector']


class Vector3d:
    def __init__(self, x, y, z):
        self._x = float(x)
        self._y = float(y)
        self._z = float(z)

    @property
    def norm(self):
        return np.sqrt(self._x**2 + self._y**2 + self._z**2)

    @property
    def unit_vector(self):
        n = self.norm
        if n > 1e-5:
            return Vector3d(self._x/n, self._y/n, self._z/n)
        else:
            return Vector3d(self._x, self._y, self._z)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    def __getitem__(self, item):
        if item == 'x' or item == 0:
            return self._x
        elif item == 'y' or item == 1:
            return self._y
        elif item == 'z' or item == 2:
            return self._z
        raise IndexError("Improper index [%s]" % item)

    def __setitem__(self, key, value):
        if key == 'x' or key == 0:
            self._x = float(value)
        elif key == 'y' or key == 1:
            self._y = float(value)
        elif key == 'z' or key == 2:
            self._z = float(value)

    def tolist(self):
        return [self._x, self._y, self._z]

    def toarray(self):
        return np.array(self.tolist(), dtype=np.float)

    def __len__(self):
        return 3

    def __add__(self, other):
        if isinstance(other, Vector3d):
            return Vector3d(self.x + other.x, self.y + other.y, self.z + other.z)
        elif (isinstance(other, list) or isinstance(other, np.array)) and len(other) == 3 and other.dtype == np.float:
            return Vector3d(self.x + other[0], self.y + other[1], self.z + other[2])
        elif isinstance(other, np.float) or isinstance(other, np.int) or isinstance(other, np.double):
            return Vector3d(self.x + other, self.y + other, self.z + other)
        raise Exception("Unsupported addition request")

    def __sub__(self, other):
        if isinstance(other, Vector3d):
            return Vector3d(self.x - other.x, self.y - other.y, self.z - other.z)
        elif (isinstance(other, list) or isinstance(other, np.array)) and len(other) == 3:
            return Vector3d(self.x - other[0], self.y - other[1], self.z - other[2])
        elif isinstance(other, np.float) or isinstance(other, np.int) or isinstance(other, np.double):
            return Vector3d(self.x - other, self.y - other, self.z - other)
        raise Exception("Unsupported substraction request")

    def __mul__(self, other):
        return Vector3d(self.x * other, self.y * other, self.z * other)

    def __str__(self):
        return "%.5fi + %.5fj + %.5fk" % (self.x, self.y, self.z)


def dotp(v1, v2):
    assert isinstance(v1, Vector3d) or (len(v1) > 0)
    assert isinstance(v2, Vector3d) or (len(v2) == len(v1))
    s = 0.0
    for i in range(len(v1)):
        s = s + v1[i] * v2[i]
    return s


def crossp(v1, v2):
    if isinstance(v1, Vector3d) and isinstance(v2, Vector3d):
        v = np.cross(v1.tolist(), v2.tolist())
        return Vector3d(v[0], v[1], v[2])
    raise Exception("cross product defined only for Vector3d object")


def connecting_vector(src, dst):
    assert isinstance(src, Coordinate3d) or len(src) == 3
    assert isinstance(dst, Coordinate3d) or len(dst) == 3
    return Vector3d(dst[0] - src[0], dst[1] - src[1], dst[2] - src[2])


def point_vector(crd):
    return Vector3d(crd.x, crd.y, crd.z)


def plane_vector(crd1, crd2, crd3):
    assert isinstance(crd1, Coordinate3d) or len(crd1) == 3
    assert isinstance(crd2, Coordinate3d) or len(crd2) == 3
    assert isinstance(crd3, Coordinate3d) or len(crd3) == 3
    v1 = connecting_vector(crd1, crd2).unit_vector
    v2 = connecting_vector(crd3, crd2).unit_vector
    return crossp(v1, v2).unit_vector
