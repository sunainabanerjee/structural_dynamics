import math
from geometry import *

__version__ = "1.0"
__all__ = ["LookupGrid"]


class LookupGrid:
    def __init__(self,
                 min_crd,
                 max_crd,
                 size,
                 max_cutoff=None,
                 min_cutoff=None):
        """
        :param min_crd: Coordinate of the lowest extreme of the box,
                        can be list, tuple, array, Coordinate3d
        :param max_crd: Coordinate of the highest extreme of the box,
                        can be list, tuple, array, Coordinate3d
        :param size: Size of the grid, grid size can be homogeneous or heterogeneous,
                     can be a list, tuple, array,
        :param max_cutoff: (optional) maximum value constraint, on any cell value
        :param min_cutoff: (optional) minimum value constraint, on any cell value
        """
        self.__max_cutoff = max_cutoff
        self.__min_cutoff = min_cutoff
        if (self.__max_cutoff is not None) and (self.__min_cutoff is not None):
            assert self.__min_cutoff < self.__max_cutoff
        """
        Setting minimum coordinate bound
        """
        if not isinstance(min_crd, Coordinate3d):
            assert len(min_crd) == 3
            min_crd = Coordinate3d(*min_crd)
        self.__min = min_crd

        """
        Setting maximum coordinate bound
        """
        if not isinstance(max_crd, Coordinate3d):
            assert len(max_crd) == 3
            max_crd = Coordinate3d(*max_crd)
        self.__max = max_crd

        """
        Setting grid cell sizes
        """
        self.__size = [1., 1., 1.]
        if hasattr(size, '__len__'):
            assert len(size) == 3
            self.__size = list(size)
        else:
            self.__size = [size, size, size]

        """
        Check data consistency
        """
        for i in range(len(self.__size)):
            assert (self.__size[i] > 1e-3) and ((self.__max[i] - self.__min[i]) // self.__size[i] > 0)

        self.__dim = [int(math.ceil((self.__max[i] - self.__min[i])/self.__size[i])) for i in range(len(self.__size))]
        self.__data = dict()

    @property
    def dim(self):
        return self.__size[0], self.__size[1], self.__size[2]

    @property
    def size(self):
        return self.__dim[0] * self.__dim[1] * self.__dim[2]

    @property
    def cell_volume(self):
        return self.__size[0] * self.__size[1] * self.__size[2]

    @property
    def volume(self):
        return self.cell_volume * self.size

    @property
    def max_coordinate(self):
        return self.__max

    @property
    def min_coordinate(self):
        return self.__min

    @property
    def max_cutoff(self):
        return self.__max_cutoff

    @property
    def min_cutoff(self):
        return self.__min_cutoff

    def __len__(self):
        return self.size

    def inside(self, x, y, z):
        crd = [x, y, z]
        return all([(crd[i] >= self.__min[i]) and (crd[i] <= self.__max[i]) for i in range(len(crd))])

    def index(self, x, y, z):
        if self.inside(x, y, z):
            crd = [x - self.__min.x, y - self.__min.y, z - self.__min.z]
            gidx = [int(crd[i] // self.__size[i]) for i in range(len(crd))]
            return gidx[0] * self.__dim[1]*self.__dim[2] + gidx[1] * self.__dim[2] + gidx[2]
        return -1

    def incr(self, key, value):
        if hasattr(key, '__len__'):
            assert len(key) == 3
            key = self.index(*key)
        if (key >= 0) and (key < self.size):
            if key in self.__data:
                value += self.__data[key]
            if (self.__max_cutoff is not None) and (value > self.__max_cutoff):
                value = self.__max_cutoff
            elif (self.__min_cutoff is not None) and (value < self.__min_cutoff):
                value = self.__min_cutoff
            self.__data[key] = value

    def __getitem__(self, item):
        if hasattr(item, '__len__'):
            assert len(item) == 3
            item = self.index(*item)
        if (item < 0) or (item > self.size):
            raise IndexError("Index out of bound!")
        return self.__data[item] if item in self.__data else 0

    def __setitem__(self, key, value):
        if hasattr(key, '__len__'):
            assert len(key) == 3
            key = self.index(*key)
        if (key >= 0) or (key < self.size):
            if (self.__max_cutoff is not None) and (value > self.__max_cutoff):
                value = self.__max_cutoff
            elif (self.__min_cutoff is not None) and (value < self.__min_cutoff):
                value = self.__min_cutoff
            self.__data[key] = value

    def __iadd__(self, key, value):
        if hasattr(key, '__len__'):
            assert len(key) == 3
            key = self.index(*key)

        if (key >= 0) or (key < self.size):
            if key in self.__data:
                value += self.__data[key]
            if (self.__max_cutoff is not None) and (value > self.__max_cutoff):
                value = self.__max_cutoff
            elif (self.__min_cutoff is not None) and (value < self.__min_cutoff):
                value = self.__min_cutoff
            self.__data[key] = value

    def reset(self):
        self.__data.clear()

    def __iter__(self):
        self.__cx = self.__min.x + 0.5 * self.__size[0]
        self.__cy = self.__min.y + 0.5 * self.__size[1]
        self.__cz = self.__min.z + 0.5 * self.__size[2]
        return self

    def __next__(self):
        cx, cy, cz = self.__cx, self.__cy, self.__cz
        if (self.__cz + self.__size[2]) < self.__max.z:
            self.__cz = self.__cz + self.__size[2]
        else:
            self.__cz = self.__min.z + 0.5 * self.__size[2]
            if (self.__cy + self.__size[1]) < self.__max.y:
                self.__cy = self.__cy + self.__size[1]
            else:
                self.__cy = self.__min.y + 0.5 * self.__size[1]
                if (self.__cx + self.__size[0]) < self.__max.x:
                    self.__cx = self.__cx + self.__size[0]
                else:
                    raise StopIteration
        return cx, cy, cz
