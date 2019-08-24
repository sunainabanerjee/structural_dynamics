import numpy as np
from geometry.vector_algebra import Coordinate3d

__version__ = "1.0"
__all__ = ['rmsd', 'kabsch_coordinate', 'kabsch_rmsd', 'rotation_matrix3d']


def rmsd(V, W):
    v = np.array(V)
    w = np.array(W)
    assert len(v) == len(w)
    return np.sqrt(np.mean(np.square(v - w)))


def kabsch_rmsd(x, y, return_matrix=False):
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert (len(x.shape) == len(y.shape)) and len(x.shape) == 2
    assert (x.shape[0] == y.shape[0]) and (x.shape[1] == y.shape[1]) and (x.shape[0] > x.shape[1])
    xc = x - x.mean(axis=0)
    yc = y - y.mean(axis=0)
    C = np.dot(np.transpose(xc), yc)
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    U = np.dot(V, W)
    xca = np.matmul(xc, U)
    r = rmsd(xca, yc)
    if not return_matrix:
        return r
    else:
        return r, xca + y.mean(axis=0)


def kabsch_coordinate(X1, X2, return_coordinate=False):
    assert isinstance(X1, list) and isinstance(X2, list)
    assert len(X1) == len(X2)
    for i in range(len(X1)):
        assert isinstance(X1[i], Coordinate3d)
        assert isinstance(X2[i], Coordinate3d)
    X1a = np.array([[X1[i].x, X1[i].y, X1[i].z] for i in range(len(X1))])
    X2a = np.array([[X2[i].x, X2[i].y, X2[i].z] for i in range(len(X2))])
    if not return_coordinate:
        return kabsch_rmsd(X1a, X2a,return_matrix=return_coordinate)
    else:
        d, X1a = kabsch_rmsd(X1a, X2a,return_matrix=return_coordinate)
        return d, [Coordinate3d(X1a[i,0], X1a[i,1], X1a[i,2]) for i in range(X1a.shape[0])]


def rotation_matrix3d(axis_vector, rotation_angle):
    assert len(axis_vector) == 3
    r = np.zeros([3, 3])
    n = np.sqrt(axis_vector[0] ** 2 + axis_vector[1] ** 2 + axis_vector[2] ** 2)
    n = 1. if n < 1e-6 else n
    ux, uy, uz = axis_vector[0] / n, axis_vector[1] / n, axis_vector[2] / n
    ct, st = np.cos(rotation_angle), np.sin(rotation_angle)

    r[0, 0] = ct + ux * ux * (1 - ct)
    r[0, 1] = ux * uy * (1 - ct) - uz * st
    r[0, 2] = ux * uz * (1 - ct) + uy * st

    r[1, 0] = ux * uy * (1 - ct) + uz * st
    r[1, 1] = ct + uy * uy * (1 - ct)
    r[1, 2] = uy * uz * (1 - ct) - ux * st

    r[2, 0] = uz * ux * (1 - ct) - uy * st
    r[2, 1] = uz * uy * (1 - ct) + ux * st
    r[2, 2] = ct + uz * uz * (1 - ct)
    return r

