import numpy as np
from geometry import Coordinate3d
from structural_dynamics import CaTrace
from .trajectory_handler import TrajectoryHandler

__version__ = "1.0"
__all__ = ['NMATrajectoryHandler']


class NMATrajectoryHandler:
    def __init__(self,
                 min_coordinate,
                 max_coordinate,
                 max_snapshot_sample=16):
        assert isinstance(max_coordinate, Coordinate3d)
        assert isinstance(min_coordinate, Coordinate3d)
        assert int(max_snapshot_sample) > 1
        self.__max_coordinate = max_coordinate
        self.__min_coordinate = min_coordinate
        self.__snapshot_sample = max_snapshot_sample
        self.__trajectory_handler = []

    def add_trajectory(self, trajectory):
        assert isinstance(trajectory, list) and len(trajectory) > 0
        assert isinstance(trajectory[0], CaTrace)
        trj_handler = TrajectoryHandler(min_coordinate=self.__min_coordinate,
                                        max_coordinate=self.__max_coordinate,
                                        max_snapshot_sample=self.__snapshot_sample)
        trj_handler.handle(trajectory=trajectory)
        trj_handler.process()
        self.__trajectory_handler.append(trj_handler)

    def reset(self):
        for trj in self.__trajectory_handler:
            trj.reset()
        self.__trajectory_handler.clear()

    def signature(self, aa_type):
        sig_list = []
        idx_list = []
        for i, trj_handler in enumerate(self.__trajectory_handler):
            sig, idx = trj_handler.signature(aa_type)
            if len(sig) > 0:
                sig_list = sig_list + sig
                idx_list = idx_list + [(i, j) for j in idx]
        return sig_list, idx_list

    def xyz(self, aa_type):
        xyz_list = []
        idx_list = []
        for i, trj_handler in enumerate(self.__trajectory_handler):
            pos_list, idx = trj_handler.xyz(aa_type)
            if len(pos_list) > 0:
                xyz_list = xyz_list + pos_list
                idx_list = idx_list + [(i, j) for j in idx]
        return xyz_list, idx_list

    def aligning_matrix(self, aa_type):
        mat_list = []
        idx_list = []
        for i, trj_handler in enumerate(self.__trajectory_handler):
            aln, idx = trj_handler.aligning_matrix(aa_type)
            if len(aln) > 0:
                mat_list = mat_list + aln
                idx_list = idx_list + [(i, j) for j in idx]
        return mat_list, idx_list

    def placement_vector(self, aa_type):
        vec_list = []
        idx_list = []
        for i, trj_handler in enumerate(self.__trajectory_handler):
            vec, idx = trj_handler.placement_vector(aa_type)
            if len(vec) > 0:
                vec_list = vec_list + vec
                idx_list = idx_list + [(i, j) for j in idx]
        return vec_list, idx_list

    @property
    def max_coordinate(self):
        return self.__max_coordinate

    @property
    def min_coordinate(self):
        return self.__min_coordinate

    def size(self, idx=None):
        if idx is None:
            return len(self.__trajectory_handler)
        return self.__trajectory_handler[idx].size

    def __len__(self):
        return self.size()

