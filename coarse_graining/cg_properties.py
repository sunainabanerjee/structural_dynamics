import numpy as np
from geometry import *
from structural_dynamics import *
from sklearn.metrics.pairwise import euclidean_distances

__version__ = "1.0"
__all__ = ['CoarseGrainProperty',
           'PropertyCalculator',
           'OneShotPropertyEstimator']


class CoarseGrainProperty:
    @staticmethod
    def sasa():
        return 'sasa'

    @staticmethod
    def volume():
        return 'volume'

    @staticmethod
    def hydropathy():
        return 'hydropathy'

    @staticmethod
    def distance():
        return 'distance'

    @staticmethod
    def angular():
        return 'angular'

    @staticmethod
    def bend():
        return 'bend'

    @staticmethod
    def flexibility():
        return 'flexibility'

    @staticmethod
    def supported_properties():
        return [CoarseGrainProperty.volume(),
                CoarseGrainProperty.sasa(),
                CoarseGrainProperty.flexibility(),
                CoarseGrainProperty.hydropathy(),
                CoarseGrainProperty.angular(),
                CoarseGrainProperty.distance(),
                CoarseGrainProperty.bend()]

    @staticmethod
    def default_properties():
        return [CoarseGrainProperty.volume(),
                CoarseGrainProperty.flexibility(),
                CoarseGrainProperty.hydropathy(),
                CoarseGrainProperty.angular(),
                CoarseGrainProperty.distance()]


class PropertyCalculator:
    @staticmethod
    def distance(ca_trace, ri, rj):
        assert isinstance(ca_trace, CaTrace)
        return distance(Coordinate3d(*ca_trace.xyz(ri)), Coordinate3d(*ca_trace.xyz(rj)))

    @staticmethod
    def angular(ca_trace, ri, rj):
        assert isinstance(ca_trace, CaTrace)
        all_residues = ca_trace.residue_ids
        assert (ri in all_residues) and (rj in all_residues)
        idx_i = all_residues.index(ri)
        assert (idx_i >= 0) and (idx_i < len(all_residues))
        ca_prev, ca_curr, ca_next = Coordinate3d(*ca_trace.xyz(all_residues[idx_i-1])),\
                                    Coordinate3d(*ca_trace.xyz(all_residues[idx_i])), \
                                    Coordinate3d(*ca_trace.xyz(all_residues[idx_i+1]))
        uv = (connecting_vector(ca_prev, ca_curr) + connecting_vector(ca_next, ca_curr)).unit_vector
        uw = connecting_vector(ca_curr, Coordinate3d(*ca_trace.xyz(rj))).unit_vector
        return dotp(uv, uw)

    @staticmethod
    def bend(ca_trace, ri):
        assert isinstance(ca_trace, CaTrace)
        all_residues = ca_trace.residue_ids
        assert (ri in all_residues)
        idx_i = all_residues.index(ri)
        if (idx_i == 0) or (idx_i == len(all_residues) - 1):
            return 1.0
        ca_prev, ca_curr, ca_next = Coordinate3d(*ca_trace.xyz(all_residues[idx_i-1])), \
                                    Coordinate3d(*ca_trace.xyz(all_residues[idx_i])), \
                                    Coordinate3d(*ca_trace.xyz(all_residues[idx_i+1]))
        uv = connecting_vector(ca_prev, ca_curr).unit_vector
        wv = connecting_vector(ca_next, ca_curr).unit_vector
        return dotp(uv, wv)

    @staticmethod
    def sasa(ca_trace, r):
        assert isinstance(ca_trace, CaTrace)
        return ca_trace.get_amino(r).sasa_free()

    @staticmethod
    def hydropathy(ca_trace, r):
        assert isinstance(ca_trace, CaTrace)
        return ca_trace.get_amino(r).hydropathy_index()

    @staticmethod
    def volume(ca_trace, r):
        assert isinstance(ca_trace, CaTrace)
        return ca_trace.get_amino(r).volume()

    @staticmethod
    def flexibility(ca_trace, r):
        assert isinstance(ca_trace, CaTrace)
        return ca_trace.get_amino(r).flexibility_index()


class OneShotPropertyEstimator:
    def __init__(self, ca_trace):
        assert isinstance(ca_trace, CaTrace)
        self.__ca_trace = ca_trace
        self.__residues = ca_trace.residue_ids
        self.__build()

    def __build(self):
        self.__xyz = np.array([[*self.__ca_trace.xyz(r)] for r in self.__residues], dtype=np.float)
        self.__allpair_distance = euclidean_distances(self.__xyz)
        u1 = self.__xyz[1:-1, :] - self.__xyz[:-2, :]
        v1 = self.__xyz[1:-1, :] - self.__xyz[2:, :]
        u1 = self.__unit_vector(u1)
        v1 = self.__unit_vector(v1)
        w = u1 + v1
        self.__ca_inplane = self.__unit_vector(w)
        self.__properties = []
        for r in self.__residues:
            aa = self.__ca_trace.get_amino(r)
            self.__properties.append({CoarseGrainProperty.sasa(): aa.sasa_free(),
                                      CoarseGrainProperty.volume(): aa.volume(),
                                      CoarseGrainProperty.flexibility(): aa.flexibility_index(),
                                      CoarseGrainProperty.hydropathy(): aa.hydropathy_index()})

    @staticmethod
    def __unit_vector(u):
        assert len(u.shape) == 2
        return (u.transpose()/np.sqrt(np.sum(np.square(u), axis=1))).transpose()

    def __neighbors(self, res_ids, k):
        assert k < (len(self.__residues) - 1)
        pidx = [self.__residues.index(r) for r in res_ids]
        return pidx, np.argsort(self.__allpair_distance[pidx], axis=-1)[:, 1:(k+1)]

    def k_neighbors(self, res_ids, k):
        if not hasattr(res_ids, '__iter__'):
            res_ids = [res_ids]
        pidx, nbrs = self.__neighbors(res_ids, k)
        return [[(self.__residues[r], self.__allpair_distance[pidx[i], r]) for r in row] for i, row in enumerate(nbrs)]

    def k_angular(self, res_ids, k):
        if not hasattr(res_ids, '__iter__'):
            res_ids = [res_ids]
        pidx, nbrs = self.__neighbors(res_ids, k)
        return [[np.dot(self.__ca_inplane[ri-1], v)
                 for v in self.__unit_vector(self.__xyz[nbrs[i]] -
                                             self.__xyz[ri])]
                for i, ri in enumerate(pidx)]

    def k_sasa(self, res_ids, k):
        if not hasattr(res_ids, '__iter__'):
            res_ids = [res_ids]
        pidx, nbrs = self.__neighbors(res_ids, k)
        return [[self.__properties[j][CoarseGrainProperty.sasa()]
                 for j in row]
                for row in nbrs]

    def k_flexibility(self, res_ids, k):
        if not hasattr(res_ids, '__iter__'):
            res_ids = [res_ids]
        pidx, nbrs = self.__neighbors(res_ids, k)
        return [[self.__properties[j][CoarseGrainProperty.flexibility()]
                 for j in row]
                for row in nbrs]

    def k_volume(self, res_ids, k):
        if not hasattr(res_ids, '__iter__'):
            res_ids = [res_ids]
        pidx, nbrs = self.__neighbors(res_ids, k)
        return [[self.__properties[j][CoarseGrainProperty.volume()]
                 for j in row]
                for row in nbrs]

    def k_hydropathy(self, res_ids, k):
        if not hasattr(res_ids, '__iter__'):
            res_ids = [res_ids]
        pidx, nbrs = self.__neighbors(res_ids, k)
        return [[self.__properties[j][CoarseGrainProperty.hydropathy()]
                 for j in row]
                for row in nbrs]

    def neighbor_signature(self,
                           res_ids,
                           k=12,
                           properties=None):
        if properties is None:
            properties = CoarseGrainProperty.default_properties()
        assert isinstance(properties, list)
        for p in properties:
            assert p in CoarseGrainProperty.default_properties()
        if not hasattr(res_ids, '__iter__'):
            res_ids = [res_ids]
        pidx, nbrs = self.__neighbors(res_ids, k)
        signature_lookup = {}
        if CoarseGrainProperty.distance() in properties:
            signature_lookup[CoarseGrainProperty.distance()] = [[self.__allpair_distance[pidx[i], r] for r in row]
                                                                for i, row in enumerate(nbrs)]
        if CoarseGrainProperty.angular() in properties:
            signature_lookup[CoarseGrainProperty.angular()] = [[np.dot(self.__ca_inplane[ri-1], v)
                                                                for v in self.__unit_vector(self.__xyz[nbrs[i]] -
                                                                                            self.__xyz[ri])]
                                                               for i, ri in enumerate(pidx)]
        if CoarseGrainProperty.hydropathy() in properties:
            signature_lookup[CoarseGrainProperty.hydropathy()] = [[self.__properties[j][CoarseGrainProperty.hydropathy()]
                                                                   for j in row]
                                                                  for row in nbrs]
        if CoarseGrainProperty.sasa() in properties:
            signature_lookup[CoarseGrainProperty.sasa()] = [[self.__properties[j][CoarseGrainProperty.sasa()]
                                                             for j in row]
                                                            for row in nbrs]
        if CoarseGrainProperty.volume() in properties:
            signature_lookup[CoarseGrainProperty.volume()] = [[self.__properties[j][CoarseGrainProperty.volume()]
                                                               for j in row]
                                                              for row in nbrs]
        if CoarseGrainProperty.flexibility() in properties:
            signature_lookup[CoarseGrainProperty.flexibility()] = [[self.__properties[j][CoarseGrainProperty.flexibility()]
                                                                   for j in row]
                                                                   for row in nbrs]
        signature = []
        for i, row in enumerate(nbrs):
            row_sig = []
            for j in range(k):
                for p in properties:
                    row_sig.append(signature_lookup[p][i][j])
            signature.append(row_sig)
        return signature





