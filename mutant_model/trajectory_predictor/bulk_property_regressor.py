import os
import numpy as np
from regressor import *
from coarse_graining import Volume3D, VolumeSignatureGrid
from mutant_model import GridPropertyLookup, LookupGrid, Signature
from mutant_model import merge_signature, SignatureAggregationOp
from structural_dynamics import valid_amino_acids
from .nma_mode_handler import NMATrajectoryHandler

__version__ = "1.0"
__all__ = ['BulkSasaRegressor', 'BulkVolumeRegressor']


def lookup_to_signature(lookup, size=None):
    assert isinstance(lookup, LookupGrid)
    sig = Signature(min_coord=lookup.min_coordinate,
                    max_coord=lookup.max_coordinate,
                    size=size)
    for x, y, z in lookup:
        sig.add(x, y, z, lookup[[x, y, z]])
    return sig


class BulkSasaRegressor:
    def __init__(self,
                 model_folder,
                 fmt_string="{}_sasa.{}",
                 model_type="xgb"):
        os.path.isdir(model_folder)
        self.__model = {}
        self.__min_sasa = 5.0
        ext = "dat" if model_type == 'xgb' else "h5"
        for aa in valid_amino_acids(one_letter=False):
            model_file = os.path.join(model_folder,
                                      fmt_string.format(aa.lower(), ext))
            assert os.path.isfile(model_file)
            if model_type == 'xgb':
                self.__model[aa] = XGBoost()
            elif model_type == 'mlp':
                self.__model[aa] = MLP()
            self.__model[aa].load(model_file)

    def signature(self, handler):
        assert isinstance(handler, NMATrajectoryHandler)
        gsize = GridPropertyLookup.sasa()
        lookup_grids = [[LookupGrid(min_crd=handler.min_coordinate,
                                    max_crd=handler.max_coordinate,
                                    size=gsize,
                                    min_cutoff=0) for j in range(handler.size(i))] for i in range(handler.size())]
        for aa in self.__model.keys():
            sig, idx = handler.signature(aa)
            if len(sig) == 0:
                continue
            sig = np.array(sig)
            sasa = self.__model[aa].predict(sig)
            pos, idx = handler.xyz(aa)
            assert (len(pos) == len(idx)) and (len(sasa) == len(idx))
            for i, item in enumerate(idx):
                sasa_value = np.round(sasa[i], decimals=1)
                if sasa_value > self.__min_sasa:
                    lookup_grids[item[0]][item[1]].incr(pos[i], sasa_value)

        sig = None
        for i in range(len(lookup_grids)):
            for j in range(len(lookup_grids[i])):
                nsig = lookup_to_signature(lookup_grids[i][j])
                if sig is None:
                    sig = nsig
                else:
                    sig = merge_signature(signature1=sig,
                                          signature2=nsig,
                                          operation=SignatureAggregationOp.sasa())
        return sig


class BulkVolumeRegressor:
    def __init__(self,
                 model_folder,
                 fmt_string="{}_volume.{}",
                 model_type="mlp"):
        os.path.isdir(model_folder)
        self.__min_occupancy = 5.0
        self.__model = {}
        ext = "dat" if model_type == 'xgb' else "h5"
        for aa in valid_amino_acids(one_letter=False):
            model_file = os.path.join(model_folder,
                                      fmt_string.format(aa.lower(), ext))
            assert os.path.isfile(model_file)
            if model_type == 'xgb':
                self.__model[aa] = XGBoost()
            elif model_type == 'mlp':
                self.__model[aa] = MLP()
            self.__model[aa].load(model_file)

    def signature(self, handler):
        assert isinstance(handler, NMATrajectoryHandler)
        gsize = GridPropertyLookup.volume()
        max_vol = gsize**3
        lookup_grids = [[LookupGrid(min_crd=handler.min_coordinate,
                                    max_crd=handler.max_coordinate,
                                    size=gsize,
                                    max_cutoff=max_vol,
                                    min_cutoff=0) for j in range(handler.size(i))] for i in range(handler.size())]
        grid = Volume3D(size=VolumeSignatureGrid.size(),
                        dim=VolumeSignatureGrid.dim())
        coordinates = np.array([[*grid.cell_center(i)] for i in range(grid.length)]).transpose()
        cell_vol = grid.cell_volume
        for aa in self.__model.keys():
            sig, idx = handler.signature(aa)
            if len(sig) == 0:
                continue
            sig = np.array(sig)
            volume = self.__model[aa].predict(sig)
            aln, idx = handler.aligning_matrix(aa)
            vec, idx = handler.placement_vector(aa)
            assert (len(aln) == len(idx)) and (len(vec) == len(idx))
            for i, item in enumerate(idx):
                coord = np.matmul(aln[i], coordinates).transpose() + vec[i].toarray()
                vol = volume[i, :] * cell_vol
                for j, crd in enumerate(coord):
                    if vol[j] > self.__min_occupancy:
                        lookup_grids[item[0]][item[1]].incr(crd, vol[j])
        vol_sig, occlusion_sig = None, None
        for i in range(len(lookup_grids)):
            for j in range(len(lookup_grids[i])):
                nsig = lookup_to_signature(lookup_grids[i][j])
                if vol_sig is None:
                    vol_sig = nsig
                    occlusion_sig = nsig
                else:
                    vol_sig = merge_signature(signature1=vol_sig,
                                              signature2=nsig,
                                              operation=SignatureAggregationOp.volume())
                    occlusion_sig = merge_signature(signature1=occlusion_sig,
                                                    signature2=nsig,
                                                    operation=SignatureAggregationOp.occlusion())
        return vol_sig, occlusion_sig



