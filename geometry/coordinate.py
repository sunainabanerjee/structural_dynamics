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
        assert (item in {0, 1, 2}) or (item in {'x', 'y', 'z'})
        if (item == 0) or (item == 'x'):
            return self._x
        elif (item == 1) or (item == 'y'):
            return self._y
        else:
            return self._z

    def __setitem__(self, key, value):
        assert (key in {0, 1, 2}) or (key in {'x', 'y', 'z'})
        assert isinstance(value, np.float)
        if (key == 0) or (key == 'x'):
            self._x = value
        elif (key == 1) or (key == 'y'):
            self._y = value
        else:
            self._z = value


def distance(coord1, coord2):
    if isinstance(coord1, Coordinate3d) and isinstance(coord2, Coordinate3d):
        return np.sqrt((coord1.x - coord2.x)**2 +
                       (coord1.y - coord2.y)**2 +
                       (coord1.z - coord2.z)**2)
    return -1


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
    x, y, z = 0, 0, 0
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)
    return x, y, z