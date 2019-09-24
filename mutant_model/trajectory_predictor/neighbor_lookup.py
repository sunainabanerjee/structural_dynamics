import time
import logging
from coarse_graining import placement_matrix, OneShotPropertyEstimator, BatchSideChainConfigHandler
from structural_dynamics import CaTrace, PDBStructure, pdb_to_catrace

__version__ = "1.0"
__all__ = ['ResidueLookup']


class ResidueLookup:
    def __init__(self, ca_trace):
        if isinstance(ca_trace, PDBStructure):
            ca_trace = pdb_to_catrace(ca_trace)
        assert isinstance(ca_trace, CaTrace)
        self.__ca_trace = ca_trace
        self.__property_estimator = OneShotPropertyEstimator(ca_trace)
        self.__sidechain_handler = BatchSideChainConfigHandler(ca_trace)
        self.__signature = {}
        self.__placement_matrix = {}
        self.__placement_vector = {}
        self.__ca_position = {}
        self.__logger = logging.getLogger("mutant_model.ResidueLookup")

    def __to_tag(self, r):
        return "R%d" % r

    def __to_res(self, t):
        assert isinstance(t, str) and t.startswith("R")
        return int(t[1:])

    def load(self, res_ids):
        start = time.time()
        self.__load_signature(res_ids)
        end = time.time()
        self.__logger.debug("Signature load time: [%.4f s]" % (end - start))

        start = time.time()
        self.__load_aligning_matrix(res_ids)
        end = time.time()
        self.__logger.debug("Aligning matrix load time: [%.4f s]" % (end - start))

        start = time.time()
        self.__load_xyz(res_ids)
        end = time.time()
        self.__logger.debug("XYZ loading time: [%.4f s]" % (end - start))

    def __load_signature(self, res_ids):
        if not hasattr(res_ids, '__iter__'):
            res_ids = [res_ids]
        res_tags = [self.__to_tag(r) for r in res_ids]
        missing = [self.__to_res(r) for r in res_tags if r not in self.__ca_position]
        if len(missing) > 0:
            sig = self.__property_estimator.neighbor_signature(missing)
            for i, r in enumerate(missing):
                self.__signature[self.__to_tag(r)] = sig[i]

    def signature(self, res_ids):
        self.load(res_ids)
        if not hasattr(res_ids, '__iter__'):
            res_ids = [res_ids]
        res_tags = [self.__to_tag(r) for r in res_ids]
        return [self.__signature[r] for r in res_tags]

    def __load_xyz(self, res_ids):
        if not hasattr(res_ids, '__iter__'):
            res_ids = [res_ids]
        res_tags = [self.__to_tag(r) for r in res_ids]
        missing = [self.__to_res(r) for r in res_tags if r not in self.__ca_position]
        if len(missing) > 0:
            for r in missing:
                self.__ca_position[self.__to_tag(r)] = self.__ca_trace.xyz(r)

    def xyz(self, res_ids):
        self.load(res_ids)
        if not hasattr(res_ids, '__iter__'):
            res_ids = [res_ids]
        res_tags = [self.__to_tag(r) for r in res_ids]
        return [self.__ca_position[r] for r in res_tags]

    def __load_aligning_matrix(self, res_ids):
        if not hasattr(res_ids, '__iter__'):
            res_ids = [res_ids]
        res_tags = [self.__to_tag(r) for r in res_ids]
        missing = [self.__to_res(r) for r in res_tags if r not in self.__placement_matrix]
        if len(missing) > 0:
            for r in missing:
                m, v = self.__sidechain_handler.placement_matrix(r)
                self.__placement_matrix[self.__to_tag(r)] = m
                self.__placement_vector[self.__to_tag(r)] = v

    def aligning_matrix(self, res_ids):
        self.load(res_ids)
        if not hasattr(res_ids, '__iter__'):
            res_ids = [res_ids]
        res_tags = [self.__to_tag(r) for r in res_ids]
        return [self.__placement_matrix[r] for r in res_tags]

    def placement_vector(self, res_ids):
        self.load(res_ids)
        if not hasattr(res_ids, '__iter__'):
            res_ids = [res_ids]
        res_tags = [self.__to_tag(r) for r in res_ids]
        return [self.__placement_vector[r] for r in res_tags]

