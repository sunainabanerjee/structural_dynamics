
__version__ = "1.0"
__all__ = ['Optimizers',
           'KernelInitializer',
           'LossFunction',
           'Activations',
           'MLPLayerType',
           'XGBooster']


class Optimizers:
    @staticmethod
    def adam():
        return 'adam'

    @staticmethod
    def rmsprop():
        return 'rmsprop'

    @staticmethod
    def sgd():
        return 'sgd'

    @staticmethod
    def supported_list(filter='nn'):
        if filter == 'nn':
            return [Optimizers.adam(),
                    Optimizers.rmsprop(),
                    Optimizers.sgd()]


class KernelInitializer():
    @staticmethod
    def he_normal():
        return 'he_normal'

    @staticmethod
    def uniform():
        return 'uniform'

    @staticmethod
    def he_uniform():
        return 'he_uniform'

    @staticmethod
    def glorot_uniform():
        return 'glorot_uniform'

    @staticmethod
    def glorot_normal():
        return 'glorot_normal'

    @staticmethod
    def supported_list():
        return [KernelInitializer.he_normal(),
                KernelInitializer.uniform(),
                KernelInitializer.he_uniform(),
                KernelInitializer.glorot_uniform(),
                KernelInitializer.glorot_normal()]


class LossFunction:
    @staticmethod
    def mean_squared_error():
        return 'mean_squared_error'

    @staticmethod
    def squared():
        return 'reg:squarederror'

    @staticmethod
    def logistic():
        return 'reg:logistic'

    @staticmethod
    def squared_log():
        return 'reg:squaredlogerror'

    @staticmethod
    def supported_list(filter='nn'):
        if filter == 'nn':
            return [LossFunction.mean_squared_error()]
        elif filter == 'xgb':
            return [LossFunction.logistic(),
                    LossFunction.squared(),
                    LossFunction.squared_log()]


class MLPLayerType:
    @staticmethod
    def dense():
        return 'dense'

    @staticmethod
    def supported_list():
        return [MLPLayerType.dense()]


class Activations:
    @staticmethod
    def relu():
        return 'relu'

    @staticmethod
    def sigmoid():
        return 'sigmoid'

    @staticmethod
    def tanh():
        return 'tanh'

    @staticmethod
    def supported_list():
        return [Activations.relu(),
                Activations.sigmoid(),
                Activations.tanh()]


class XGBooster:
    @staticmethod
    def gbtree():
        return 'gbtree'

    @staticmethod
    def dart():
        return 'dart'

    @staticmethod
    def gblinear():
        return 'gblinear'

    @staticmethod
    def supported_list():
        return [XGBooster.gbtree(),
                XGBooster.dart(),
                XGBooster.gblinear()]
