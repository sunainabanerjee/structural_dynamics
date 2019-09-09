import math
from .coordinate import *

__version__ = "1.0"
__all__ = ['cuboid_volume',
           'sphere_volume',
           'sphere_intersection_volume',
           'equivalent_sphere_radius']


def cuboid_volume(sx, sy, sz):
    assert (sx >= 0) and (sy >= 0) and (sz >= 0)
    return sx * sy * sz


def sphere_volume(r):
    assert r >= 0
    return 4.0 * math.pi * (r**3) / 3.0


def sphere_intersection_volume(r1, center1, r2, center2):
    assert isinstance(center1, Coordinate3d) and r1 > 0
    assert isinstance(center2, Coordinate3d) and r2 > 0
    d = distance(center1, center2)
    if d > (r1 + r2):
        return 0
    if d < abs(r1 - r2):
        return sphere_volume(min(r1, r2))
    f = (d**2 + 2*d*(r1 + r2) - 3*(r1**2 + r2**2) + 6*r1*r2) / (12*d)
    return math.pi * ((r1 + r2 - d)**2) * f


def equivalent_sphere_radius(vol):
    assert vol >= 0.
    return ((vol * 3)/(4 * math.pi))**(1.0/3.0)

