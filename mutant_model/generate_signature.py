import logging
import time
import numpy as np
from geometry import *
from .alignment import *
from .signature import *
from .lookup_grid import *
from .system_constants import *
from .prediction_wrapper import *
from structural_dynamics import *

__version__ = "1.0"
__all__ = ['SignatureGenerator', 'trajectory_signature']


class SignatureGenerator:
    def __init__(self,
                 min_coordinate,
                 max_coordinate,
                 properties=None):
        if properties is None:
            properties = DiscriminantProperties.all_properties()
        assert isinstance(min_coordinate, Coordinate3d)
        assert isinstance(max_coordinate, Coordinate3d)
        assert all([p in DiscriminantProperties.all_properties() for p in properties])
        limiting_p = np.max([GridPropertyLookup.grid_size(p) for p in properties])
        self.__lookup = LookupGrid(min_crd=min_coordinate,
                                   max_crd=max_coordinate,
                                   size=limiting_p)
        self.__eval = {}
        for p in properties:
            self.__eval[p] = get_predictor(p, min_coordinate, max_coordinate)
        self.__logger = logging.getLogger("mutant_model.SignatureGenerator")

    @property
    def assigned_properties(self):
        return sorted(list(self.__eval.keys()))

    def signature(self, ca_trace):
        assert isinstance(ca_trace, CaTrace)
        all_residues = ca_trace.residue_ids
        rlist = list()
        for r in all_residues:
            if self.__lookup.inside(*ca_trace.xyz(r)):
                rlist.append(r)
        assert len(rlist) > 0
        all_signature = {}
        for p in self.__eval.keys():
            all_signature[p] = Signature(min_coord=self.__lookup.min_coordinate,
                                         max_coord=self.__lookup.max_coordinate)
            self.__logger.debug("Generating signature for: %s" % p)
            print("Generating signature for: %s" % p)  #summukhe
            start = time.time()
            self.__eval[p].reset()
            self.__eval[p].add(ca_trace)
            for x, y, z in self.__eval[p]:
                all_signature[p].add(x, y, z, self.__eval[p].estimate(x, y, z))
            end = time.time()
            etime = (end - start)*1000
            self.__logger.debug("Feature [%s] generation time %.2f ms" % (p, etime))
            print("Feature [%s] generation time %.2f ms" % (p, etime)) #summukhe
        return all_signature


def trajectory_signature(ref_structure,
                         trajectory,
                         sig_gen,
                         max_snapshot=5,
                         align_pair=None):
    assert isinstance(ref_structure, CaTrace)
    assert isinstance(sig_gen, SignatureGenerator)
    assert isinstance(trajectory, list) and len(trajectory) > 0
    assert isinstance(trajectory[0], CaTrace)
    assert max_snapshot > 0
    logger = logging.getLogger("trajectory_signature")

    if len(trajectory) > max_snapshot:
        dstep = int(len(trajectory) // max_snapshot)
        trajectory = [snapshot for i, snapshot in enumerate(trajectory) if i % dstep == 0]

    start = time.time()
    trajectory = align_trajectory(ref_structure,
                                  trajectory[0],
                                  trajectory,
                                  align_pair=align_pair,
                                  ca_trace=True)
    end = time.time()
    etime = (end - start)*1000
    logger.debug("Aligned trajectory to reference : time [%.2f ms]" % etime)
    print("Aligned trajectory to reference : time [%.2f ms]" % etime)

    properties = sig_gen.assigned_properties
    sig = {p: None for p in properties}
    for trace in trajectory:
        start = time.time()
        sp = sig_gen.signature(trace)
        end = time.time()
        etime = (end - start)*1000
        logger.debug("Signature generation time [%.2f ms]" % etime)
        print("Signature generation time [%.2f ms]" % etime)  #summukhe
        for p in properties:
            if sig[p] is None:
                sig[p] = sp[p]
            else:
                sig[p] = merge_signature(sig[p],
                                         sp[p],
                                         operation=SignatureAggregationOp.get_operation(p))
    return sig

