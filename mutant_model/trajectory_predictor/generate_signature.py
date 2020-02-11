import os
from .nma_mode_handler import *
from mutant_model import signature_diff
from .bulk_property_regressor import *
from mutant_model import GridPropertyLookup

__version__ = "1.0"
__all__ = ['BulkSignatureGenerator']


class BulkSignatureGenerator:
    def __init__(self,
                 sasa_model='xgb',
                 volume_model='mlp',
                 model_folder=None):
        if model_folder is None:
            model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                        '..', 'models')
        assert os.path.isdir(model_folder)
        sasa_folder = os.path.join(model_folder, 'sasa_models')
        volume_folder = os.path.join(model_folder, 'volume_models')
        self.__sasa_model = BulkSasaRegressor(model_folder=sasa_folder,
                                              model_type=sasa_model)
        self.__volume_model = BulkVolumeRegressor(model_folder=volume_folder,
                                                  model_type=volume_model)

    def signature(self, wild_handler, mutant_handler):
        assert isinstance(wild_handler, NMATrajectoryHandler)
        assert isinstance(mutant_handler, NMATrajectoryHandler)
        assert wild_handler.size() == mutant_handler.size()

        wild_sasa_sig = self.__sasa_model.signature(wild_handler)
        mutant_sasa_sig = self.__sasa_model.signature(mutant_handler)

        wild_vol_sig, wild_occ_sig = self.__volume_model.signature(wild_handler)
        mut_vol_sig, mut_occ_sig = self.__volume_model.signature(mutant_handler)

        sasa_score = signature_diff(mutant_sasa_sig, wild_sasa_sig)
        vol_score = signature_diff(mut_vol_sig, wild_vol_sig)
        occ_score = signature_diff(mut_occ_sig, wild_occ_sig)
        sig = [m for m, s in sasa_score] + [s for m, s in sasa_score] +\
              [m for m, s in vol_score] + [s for m, s in vol_score] +\
              [m for m, s in occ_score] + [s for m, s in occ_score]
        assert len(sig) == BulkSignatureGenerator.size()
        return sig

    @staticmethod
    def size():
        sz_x, sz_y, sz_z = GridPropertyLookup.signature_grid_dim()
        return 6 * (sz_x - 1) * (sz_y - 1) * (sz_z - 1)

    @staticmethod
    def dim_names():
        sz_x, sz_y, sz_z = GridPropertyLookup.signature_grid_dim()
        property_order = ('sasa', 'volume', 'occlusion')
        sub_order = ('mean', 'variance')
        names = []
        for p in property_order:
            names = names + ['%s_%s(%d,%d,%d)' % (p, o, x, y, z)
                             for o in sub_order
                             for x in range(1, sz_x)
                             for y in range(1, sz_y)
                             for z in range(1, sz_z)]
        return names

