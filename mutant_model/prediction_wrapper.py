import os
import logging
from abc import ABC
from .lookup_grid import *
from .system_constants import *
from geometry import *
from coarse_graining import *
from abc import abstractmethod
from structural_dynamics import *

__version__ = "1.0"
__all__ = ['PropertyEstimator',
           'VolumeEstimator',
           'ElectrostaticEstimator',
           'SASAEstimator',
           'get_predictor']


class PropertyEstimator(ABC):
    @abstractmethod
    def dim(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def set(self,
            min_crd,
            max_crd,
            size=None):
        pass

    @abstractmethod
    def add(self, ca_trace):
        pass

    @abstractmethod
    def estimate(self, x, y, z):
        pass

    @abstractmethod
    def __iter__(self):
        pass


def get_predictor(prop, min_coordinate, max_coordinate):
    assert prop in DiscriminantProperties.all_properties()
    model = None
    if prop == DiscriminantProperties.sasa():
        model_folder = os.path.join( os.path.dirname(os.path.realpath(__file__)), 'models', 'sasa_models')
        assert os.path.isdir(model_folder)
        model = SASAEstimator(model_folder=model_folder)
    elif prop == DiscriminantProperties.volume():
        model_folder = os.path.join( os.path.dirname(os.path.realpath(__file__)), 'models', 'volume_models')
        assert os.path.isdir(model_folder)
        model = VolumeEstimator(model_folder=model_folder)
    elif prop == DiscriminantProperties.electrostatic():
        model_folder = os.path.join( os.path.dirname(os.path.realpath(__file__)), 'models', 'pos_models')
        assert os.path.isdir(model_folder)
        model = ElectrostaticEstimator(model_folder=model_folder)
    elif prop == DiscriminantProperties.occlusion():
        model_folder = os.path.join( os.path.dirname(os.path.realpath(__file__)), 'models', 'volume_models')
        assert os.path.isdir(model_folder)
        model = VolumeEstimator(model_folder=model_folder)
    model.set(min_coordinate, max_coordinate)
    return model


class VolumeEstimator(PropertyEstimator, ABC):
    def __init__(self,
                 model_folder,
                 fmt_string="{}_volume.{}",
                 model_type='mlp'):
        assert os.path.isdir(model_folder)
        assert model_type in {'xgb', 'mlp'}
        self.__model = VolumePredictor(model_folder=model_folder,
                                       fmt_string=fmt_string,
                                       model_type=model_type)
        self.__lookup = None
        self.__logger = logging.getLogger("mutant_model.VolumeEstimator")

    def dim(self):
        return self.__lookup.dim

    def set(self, min_crd, max_crd, size=None):
        if size is None:
            size = GridPropertyLookup.volume()

        if hasattr(size, '__len__'):
            max_cutoff = size[0] * size[1] * size[2]
        else:
            max_cutoff = size**3

        self.__lookup = LookupGrid(min_crd=min_crd,
                                   max_crd=max_crd,
                                   size=size,
                                   max_cutoff=max_cutoff,
                                   min_cutoff=0)

    def reset(self):
        if self.__lookup is not None:
            self.__lookup.reset()

    def add(self, ca_trace):
        assert isinstance(ca_trace, CaTrace)
        assert self.__lookup is not None
        resids = []
        for r in ca_trace.residue_ids[1:-1]:
            if self.__lookup.inside(*ca_trace.xyz(r)):
                resids.append(r)
        if len(resids) > 0:
            recons = self.__model.predict_volume(ca_trace=ca_trace,
                                                 residue_ids=resids)
            for r, coord, vol in recons:
                n = coord.shape[0]
                for i in range(n):
                    if self.__lookup.inside(*coord[i, :]):
                        self.__lookup[coord[i, :]] += vol[i]

    def estimate(self, x, y, z):
        assert self.__lookup is not None
        return self.__lookup[Coordinate3d(x, y, z)]

    def __iter__(self):
        assert self.__lookup is not None
        return iter(self.__lookup)


class ElectrostaticEstimator(PropertyEstimator, ABC):
    def __init__(self,
                 model_folder,
                 fmt_string="{}_pos.{}",
                 model_type='xgb'):
        assert os.path.isdir(model_folder)
        assert model_type in {'xgb', 'mlp'}
        self.__model = ElectrostaticModel(model_folder=model_folder,
                                          fmt_string=fmt_string,
                                          model_type=model_type)
        self.__lookup = None

    def dim(self):
        return GridPropertyLookup.electrostatic(), \
               GridPropertyLookup.electrostatic(), \
               GridPropertyLookup.electrostatic()

    def set(self, min_crd, max_crd, size=None):
        if size is None:
            size = GridPropertyLookup.volume()
        self.__lookup = LookupGrid(min_crd=min_crd,
                                   max_crd=max_crd,
                                   size=size)

    def add(self, ca_trace):
        assert self.__lookup is not None
        self.__model.add_protein(ca_trace)

    def estimate(self, x, y, z):
        return self.__model.potential(x, y, z)

    def reset(self):
        return self.__model.reset()

    def __iter__(self):
        assert self.__lookup is not None
        return iter(self.__lookup)


class SASAEstimator(PropertyEstimator, ABC):
    def __init__(self,
                 model_folder,
                 fmt_string="{}_sasa.{}",
                 model_type='xgb'):
        assert os.path.isdir(model_folder)
        assert model_type in {'xgb', 'mlp'}
        self.__model = SASAPredictor(model_folder=model_folder,
                                     fmt_string=fmt_string,
                                     model_type=model_type)
        self.__lookup = None

    def dim(self):
        if self.__lookup is not None:
            return self.__lookup.dim
        return None

    def set(self, min_crd, max_crd, size=None):
        if size is None:
            size = GridPropertyLookup.volume()

        self.__lookup = LookupGrid(min_crd=min_crd,
                                   max_crd=max_crd,
                                   size=size,
                                   min_cutoff=0)

    def add(self, ca_trace):
        assert isinstance(ca_trace, CaTrace)
        assert self.__lookup is not None
        resids = []
        all_residues = ca_trace.residue_ids
        all_residues = [r for r in all_residues if (r > 0) and
                        (all_residues.index(r) > 1) and
                        (all_residues.index(r) < len(all_residues)-1)]
        for r in all_residues:
            if self.__lookup.inside(*ca_trace.xyz(r)):
                resids.append(r)

        if len(resids) > 0:
            r, sasa = self.__model.predict_sasa(ca_trace=ca_trace,
                                                resids=resids)
            for i, r in enumerate(r):
                coord = Coordinate3d(*ca_trace.xyz(r))
                self.__lookup[coord] += sasa[i]

    def estimate(self, x, y, z):
        if self.__lookup is not None:
            return self.__lookup[Coordinate3d(x, y, z)]
        return 0

    def reset(self):
        if self.__lookup is not None:
            self.__lookup.reset()

    def __iter__(self):
        assert self.__lookup is not None
        return iter(self.__lookup)

