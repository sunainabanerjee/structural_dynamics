import numpy as np
from .coordinate import *

__version__ = "1.0"
__all__ = ['Vector3d', 'dotp', 'crossp',
           'connecting_vector', 'point_vector']


class Vector3d:
    def __init__(self, x, y, z):
        assert isinstance(x, np.float)
        assert isinstance(y, np.float)
        assert isinstance(z, np.float)
        self._x = x
        self._y = y
        self._z = z

    @property
    def norm(self):
        return np.sqrt(self._x**2 + self._y**2 + self._z**2)

    @property
    def unit_vector(self):
        n = self.norm
        if n > 1e-3:
            return Vector3d( self._x/n , self._y/n, self._z/n)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    def tolist(self):
        return [self._x, self._y, self._z]

    def toarray(self):
        return np.array(self.tolist())

    def __add__(self, other):
        if isinstance(other, Vector3d):
            return Vector3d(self._x + other.x, self._y + other.y, self._z + other.z)
        elif (isinstance(other, list) or isinstance(other, np.array)) and len(other) == 3 and other.dtype == np.float:
            return Vector3d(self._x + other[0], self._y + other[1], self._z + other[2])
        elif isinstance(other, np.float) or isinstance(other, np.int) or isinstance(other, np.double):
            return Vector3d(self._x + other, self._y + other, self._z + other)
        raise Exception("Unsupported addition request")

    def __sub__(self, other):
        if isinstance(other, Vector3d):
            return Vector3d(self._x - other.x, self._y - other.y, self._z - other.z)
        elif (isinstance(other, list) or isinstance(other, np.array)) and len(other) == 3 and other.dtype == np.float:
            return Vector3d(self._x - other[0], self._y - other[1], self._z - other[2])
        elif isinstance(other, np.float) or isinstance(other, np.int) or isinstance(other, np.double):
            return Vector3d(self._x - other, self._y - other, self._z - other)
        raise Exception("Unsupported substraction request")


def dotp(v1, v2):
    if isinstance(v1, Vector3d) and isinstance(v2, Vector3d):
        return (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z)
    return None


def crossp(v1, v2):
    if isinstance(v1, Vector3d) and isinstance(v2, Vector3d):
        v = np.cross(v1.tolist(), v2.tolist())
        return Vector3d(v[0], v[1], v[2])
    raise Exception("cross product defined only for Vector3d object")


def connecting_vector(src, dst):
    assert isinstance(src, Coordinate3d) and isinstance(dst, Coordinate3d)
    return Vector3d(dst.x - src.x, dst.y - src.y, dst.z - src.z)


def point_vector(crd):
    return Vector3d(crd.x, crd.y, crd.z)






