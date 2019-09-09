from geometry import *
from structural_dynamics import *

__version__ = "1.0"
__all__ = ['CoarseGrainProperty', 'PropertyCalculator']


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

