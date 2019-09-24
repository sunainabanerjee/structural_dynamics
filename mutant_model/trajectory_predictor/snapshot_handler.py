import time
import logging
from .neighbor_lookup import ResidueLookup
from structural_dynamics import CaTrace, PDBStructure, get_amino

__version__ = "1.0"
__all__ = ["SnapshotHandler"]


class SnapshotHandler:
    def __init__(self, ca_trace):
        assert isinstance(ca_trace, (CaTrace, PDBStructure))
        self.__residue_lookup = ResidueLookup(ca_trace=ca_trace)
        self.__ca_trace = ca_trace
        self.__model_track = {}
        self.__pos_lookup = {}
        self.__logger = logging.getLogger("mutant_model.SnapshotHandler")

    def process(self, residue_ids):
        start = time.time()
        self.__residue_lookup.load(residue_ids)
        aa_order = [self.__ca_trace.get_amino(r).name(one_letter_code=False) for r in residue_ids]
        for i, aa in enumerate(aa_order):
            if aa not in self.__model_track:
                self.__model_track[aa] = set()
            self.__model_track[aa].add(residue_ids[i])
        end = time.time()
        self.__logger.debug("Snapshot handler process time: %.3f s" % (end - start))

    def residue_ids(self, aa_type):
        aa = get_amino(aa_type).name(one_letter_code=False)
        if aa in self.__model_track:
            return sorted(self.__model_track[aa])
        return []

    def signature(self, aa_type):
        aa = get_amino(aa_type).name(one_letter_code=False)
        if aa in self.__model_track:
            res_ids = sorted(self.__model_track[aa])
            return self.__residue_lookup.signature(res_ids)
        return None

    def xyz(self, aa_type):
        aa = get_amino(aa_type).name(one_letter_code=False)
        if aa in self.__model_track:
            res_ids = sorted(self.__model_track[aa])
            return self.__residue_lookup.xyz(res_ids)
        return None

    def aligning_matrix(self, aa_type):
        aa = get_amino(aa_type).name(one_letter_code=False)
        if aa in self.__model_track:
            res_ids = sorted(self.__model_track[aa])
            return self.__residue_lookup.aligning_matrix(res_ids)
        return None

    def placement_vector(self, aa_type):
        aa = get_amino(aa_type).name(one_letter_code=False)
        if aa in self.__model_track:
            res_ids = sorted(self.__model_track[aa])
            return self.__residue_lookup.placement_vector(res_ids)
        return None

