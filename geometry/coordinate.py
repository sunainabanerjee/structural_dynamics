import math
import numpy as np

__version__ = "1.0"
__all__ = ['Coordinate3d', 'distance', 'minimum_bound', 'maximum_bound',
           'cartesian_to_spherical', 'spherical_to_cartesian']


class Coordinate3d:
    def __init__(self, x, y, z):
        self._x = x
        self._y = y
        self._z = z

    def __str__(self):
        return '(%8.3f, %8.3f, %8.3f)' % (self._x, self._y, self._z)

    def __len__(self):
        return 3

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
        if (item == 0) or (item == 'x'):
            return self._x
        elif (item == 1) or (item == 'y'):
            return self._y
        elif (item == 2) or (item == 'z'):
            return self._z
        raise IndexError("Invalid index [%s]" % item)

    def __setitem__(self, key, value):
        value = float(value)
        if (key == 0) or (key == 'x'):
            self._x = value
        elif (key == 1) or (key == 'y'):
            self._y = value
        elif (key == 2) or (key == 'z'):
            self._z = value
        else:
            raise IndexError("Invalid index [%s]" % key)

    def tolist(self):
        return [self._x, self._y, self._z]

    def toarray(self):
        return np.array([self._x, self._y, self._z], dtype=np.float)

    @property
    def values(self):
        return self._x, self._y, self._z


def distance(c1, c2):
    assert isinstance(c1, Coordinate3d) or (len(c1) == 3)
    assert isinstance(c2, Coordinate3d) or (len(c2) == 3)
    s = 0
    for i in range(len(c1)):
        s += (c1[i] - c2[i])**2
    return np.sqrt(s)


def minimum_bound(coordinates):
    assert isinstance(coordinates, list) and (len(coordinates) > 0)
    x, y, z = coordinates[0].x, coordinates[0].y, coordinates[0].z
    for crd in coordinates:
        if x > crd.x:
            x = crd.x
        if y > crd.y:
            y = crd.y
        if z > crd.z:
            z = crd.z
    return x, y, z


def maximum_bound(coordinates):
    assert isinstance(coordinates, list) and (len(coordinates) > 0)
    x, y, z = coordinates[0].x, coordinates[0].y, coordinates[0].z
    for crd in coordinates:
        if x < crd.x:
            x = crd.x
        if y < crd.y:
            y = crd.y
        if z < crd.z:
            z = crd.z
    return x, y, z


def cartesian_to_spherical(x, y, z):
    r, theta, phi = 0, 0, 0
    r = math.sqrt(x**2 + y**2 + z**2)
    if r > 0:
        theta = (2 * math.pi + math.acos(z/r)) % (2*math.pi)
        phi = (2 * math.pi + math.atan2(y, x)) % (2*math.pi)
    return r, theta, phi


def spherical_to_cartesian(r, theta, phi):
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)
    return x, y, z