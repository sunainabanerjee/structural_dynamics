
__version__ = "1.0"
__all__ = ['DiscriminantProperties',
           'GridPropertyLookup',
           'SignatureAggregationOp']


class DiscriminantProperties:
    @staticmethod
    def sasa():
        return 'sasa'

    @staticmethod
    def electrostatic():
        return 'electrostatic'

    @staticmethod
    def volume():
        return 'volume'

    @staticmethod
    def occlusion():
        return 'occlusion'

    @staticmethod
    def all_properties():
        return [DiscriminantProperties.sasa(),
                DiscriminantProperties.volume(),
                DiscriminantProperties.occlusion()]


class SignatureAggregationOp:
    @staticmethod
    def sasa():
        return 'concat'

    @staticmethod
    def electrostatic():
        return 'concat'

    @staticmethod
    def volume():
        return 'concat'

    @staticmethod
    def occlusion():
        return 'max'

    @staticmethod
    def get_operation(prop):
        assert prop in DiscriminantProperties.all_properties()
        if prop == DiscriminantProperties.sasa():
            return SignatureAggregationOp.sasa()
        elif prop == DiscriminantProperties.volume():
            return SignatureAggregationOp.volume()
        elif prop == DiscriminantProperties.electrostatic():
            return SignatureAggregationOp.electrostatic()
        elif prop == DiscriminantProperties.occlusion():
            return SignatureAggregationOp.occlusion()


class GridPropertyLookup:
    @staticmethod
    def grid_size(prop):
        assert prop in DiscriminantProperties.all_properties()
        if prop == DiscriminantProperties.sasa():
            return GridPropertyLookup.sasa()
        elif prop == DiscriminantProperties.occlusion():
            return GridPropertyLookup.occlusion()
        elif prop == DiscriminantProperties.volume():
            return GridPropertyLookup.volume()
        elif prop == DiscriminantProperties.electrostatic():
            return GridPropertyLookup.electrostatic()

    def sasa():
        return 9.5

    @staticmethod
    def electrostatic():
        return 2.5

    @staticmethod
    def volume():
        return 9.5

    @staticmethod
    def occlusion():
        return 9.5

    @staticmethod
    def signature_grid_dim():
        return 4, 4, 4
