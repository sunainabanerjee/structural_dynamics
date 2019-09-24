import time
import logging
import numpy as np
from geometry import Coordinate3d
from structural_dynamics import CaTrace
from mutant_model import DiscriminantProperties, GridPropertyLookup, LookupGrid
from .snapshot_handler import SnapshotHandler

__version__ = "1.0"
__all__ = ['TrajectoryHandler']


class TrajectoryHandler:
    def __init__(self,
                 min_coordinate,
                 max_coordinate,
                 max_snapshot_sample=5):
        assert max_snapshot_sample > 1
        assert isinstance(min_coordinate, Coordinate3d)
        assert isinstance(max_coordinate, Coordinate3d)
        self.__max_sample = int(max_snapshot_sample)
        self.__max_coordinate = max_coordinate
        self.__min_coordinate = min_coordinate
        smallest_side = np.min([self.__max_coordinate[i] - self.__min_coordinate[i] for i in range(len(self.__max_coordinate))])
        self.__lookup_grid = LookupGrid(min_crd=min_coordinate,
                                        max_crd=max_coordinate,
                                        size=smallest_side / 2.0)
        self.__snapshot_handlers = []
        self.__residue_set = set()
        self.__logger = logging.getLogger("mutant_model.TrajectoryHandler")

    def handle(self, trajectory):
        assert isinstance(trajectory, list) and len(trajectory) > 0
        assert isinstance(trajectory[0], CaTrace)
        n = len(trajectory)
        if n > self.__max_sample:
            dstep = int(n // self.__max_sample)
            trajectory = [snapshot for i, snapshot in enumerate(trajectory) if i % dstep == 0]
        assert len(trajectory) > 0
        for snapshot in trajectory:
            all_residues = snapshot.residue_ids[1:-1]
            for r in all_residues:
                if self.__lookup_grid.inside(*snapshot.xyz(r)):
                    self.__residue_set.add(r)
            self.__snapshot_handlers.append(SnapshotHandler(ca_trace=snapshot))

    def process(self):
        assert len(self.__residue_set) > 0
        residue_list = sorted(self.__residue_set)
        start = time.time()
        for h in self.__snapshot_handlers:
            h.process(residue_list)
        end = time.time()
        self.__logger.debug("Process time for [%d] handler: %.3f" % (len(self.__snapshot_handlers), (end - start)))

    @property
    def residue_ids(self):
        return sorted(self.__residue_set)

    def reset(self):
        self.__residue_set.clear()
        self.__snapshot_handlers.clear()

    def xyz(self, aa_type):
        snapshot_ids = []
        xyz_list = []
        for i, h in enumerate(self.__snapshot_handlers):
            res = h.xyz(aa_type=aa_type)
            if res is not None:
                n = len(res)
                xyz_list = xyz_list + res
                snapshot_ids = snapshot_ids + [i]*n
        return xyz_list, snapshot_ids

    def aligning_matrix(self, aa_type):
        snapshot_ids = []
        aln_list = []
        for i, h in enumerate(self.__snapshot_handlers):
            res = h.aligning_matrix(aa_type=aa_type)
            if res is not None:
                n = len(res)
                aln_list = aln_list + res
                snapshot_ids = snapshot_ids + [i]*n
        return aln_list, snapshot_ids

    def signature(self, aa_type):
        snapshot_ids = []
        sig_list = []
        for i, h in enumerate(self.__snapshot_handlers):
            res = h.signature(aa_type=aa_type)
            if res is not None:
                n = len(res)
                sig_list = sig_list + res
                snapshot_ids = snapshot_ids + [i]*n
        return sig_list, snapshot_ids

    def placement_vector(self, aa_type):
        snapshot_ids = []
        vec_list = []
        for i, h in enumerate(self.__snapshot_handlers):
            res = h.placement_vector(aa_type=aa_type)
            if res is not None:
                n = len(res)
                vec_list = vec_list + res
                snapshot_ids = snapshot_ids + [i]*n
        return vec_list, snapshot_ids

    @property
    def size(self):
        return len(self.__snapshot_handlers)

    def __len__(self):
        return self.size
