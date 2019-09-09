import os
import re
import fnmatch
import numpy as np
from .mlp import MLP
from .utility import *
from .regressor import Regressor

__version__ = "1.0"
__all__ = ['ChainedMLP']


class ChainedMLP(Regressor):
    def __init__(self,
                 chain_length,
                 chained_incr=3,
                 nepochs=2500,
                 batch_size=200,
                 optimizer=Optimizers.adam(),
                 kernel_initializer=KernelInitializer.glorot_normal(),
                 loss=LossFunction.mean_squared_error()):
        assert optimizer in Optimizers.supported_list()
        assert kernel_initializer in KernelInitializer.supported_list()
        assert loss in LossFunction.supported_list()
        self.__name = 'chained_nn'
        assert (int(chain_length) > 0) and (int(chained_incr) > 0)
        assert (int(nepochs) > 50) and (int(batch_size) > 0)
        self.__chain_length = int(chain_length)
        self.__epochs = int(nepochs)
        self.__batch_size = int(batch_size)
        self.__optimizer = optimizer
        self.__lr = 1e-3
        self.__input_dim = None
        self.__output_dim = int(chained_incr)
        self.__initializer = kernel_initializer
        self.__loss_function = loss
        self.__model = [MLP(epochs=self.__epochs,
                            batch_size=batch_size,
                            optimizer=optimizer,
                            kernel_initializer=kernel_initializer) for i in range(self.__chain_length)]
        for i in range(len(self.__model)):
            self.__model[i].set_learning_rate(self.__lr)
            self.__model[i].set_output_shape(self.__output_dim)

    @property
    def name(self):
        return self.__name

    def __len__(self):
        return self.__chain_length

    def __getitem__(self, item):
        assert (item < len(self.__model)) and item >= 0
        return self.__model[item]

    @property
    def nepochs(self):
        return self.__epochs

    def set_epochs(self, epochs):
        if (epochs > 0) and int(epochs) != self.__epochs:
            self.__epochs = int(epochs)
            for i in range(self.__chain_length):
                self.__model[i].set_epochs(self.__epochs)

    @property
    def optimizer(self):
        return self.__optimizer

    def set_optimizer(self, opt):
        assert opt in Optimizers.supported_list(filter='nn')
        self.__optimizer = opt
        for i in range(self.__chain_length):
            self.__model[i].set_optimizer(opt)

    @property
    def learning_rate(self):
        return self.__lr

    def set_learning_rate(self, lr):
        assert (lr > 0) and (lr <= 1.0)
        self.__lr = lr
        for i in range(self.__chain_length):
            self.__model[i].set_learning_rate(self.__lr)

    @property
    def input_shape(self):
        return self.__input_dim

    def set_input_shape(self, input_dim):
        assert (self.__input_dim is None) and (input_dim > 0)
        if self.__output_dim is not None:
            assert int(input_dim) > self.output_shape
        self.__input_dim = int(input_dim)
        for i in range(self.__chain_length):
           self.__model[i].set_input_shape(self.__input_dim + i * self.output_shape)

    @property
    def output_shape(self):
        return self.__output_dim

    @property
    def kernel_initializer(self):
        return self.__initializer

    def set_kernel_initializer(self, initializer):
        assert initializer in KernelInitializer.supported_list()
        self.__initializer = initializer
        for i in range(len(self.__model)):
            self.__model[i].set_kernel_initializer(initializer)

    @property
    def batch_size(self):
        return self.__batch_size

    def set_batch_size(self, size):
        if size > 0:
            self.__batch_size = int(size)
        for i in range(len(self.__model)):
            self.__model[i].set_batch_size(size)

    @property
    def loss_function(self):
        return self.__loss_function

    def set_loss(self, loss_function):
        assert loss_function in LossFunction.supported_list()
        self.__loss_function = loss_function
        for i in range(len(self.__model)):
            self.__model[i].set_loss(self.__loss_function)

    @property
    def nlayers(self):
        return self.__model[0].nlayers

    def add_layers(self,
                   dim,
                   layer_type=MLPLayerType.dense(),
                   activation=Activations.relu()):
        assert layer_type in MLPLayerType.supported_list()
        assert activation in Activations.supported_list()
        for i in range(len(self.__model)):
            self.__model[i].add_layers(dim=dim,
                                       layer_type=layer_type,
                                       activation=activation)

    def set_model(self):
        assert self.input_shape is not None
        assert self.output_shape is not None
        assert (self.nlayers > 0)
        for i in range(len(self.__model)):
            self.__model[i].set_model()

    def fit(self, x, y, test_split=0.2):
        assert (test_split > 0) and (test_split <= 0.5)
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
        assert (len(x.shape) == 2) and (len(y.shape) <= 2)
        assert (x.shape[0] == y.shape[0])
        assert (x.shape[1] == self.input_shape)
        assert (y.shape[1] == self.output_shape * self.__chain_length)
        accuracies = []
        for i in range(self.__chain_length):
            first_idx = i * self.output_shape
            last_idx = (i + 1) * self.output_shape
            y_data = y[:, first_idx:last_idx]
            if i > 0:
                x_data = np.concatenate((x, y[:, :first_idx]), axis=1)
            else:
                x_data = x.copy()
            acc = self.__model[i].fit(x_data, y_data, test_split=test_split)
            accuracies.append(acc)
        return accuracies

    def predict(self, x):
        assert self.__model is not None
        inp = x
        result = None
        for i in range(self.__chain_length):
            y = self.__model[i].predict(inp)
            inp = np.concatenate((inp, y), axis=1)
            result = y if result is None else np.concatenate((result, y), axis=1)
        return result

    @property
    def has_training_history(self):
        return all([self.__model[i].has_training_history for i in range(self.__chain_length)])

    def train_history(self, item):
        if item < self.__chain_length:
            return self.__model[item].train_history

    def save(self, file_path):
        if self.has_training_history:
            if not os.path.isdir(os.path.realpath(file_path)):
                os.mkdir(os.path.realpath(file_path))
                for i in range(self.__chain_length):
                    fname = os.path.join(os.path.realpath(file_path), 'model_%d.h5' % (i+1))
                    self.__model[i].save(fname)

    def load(self, model_path):
        assert os.path.isdir(model_path)
        models = {}
        for f in fnmatch.filter(os.listdir(model_path), 'model_[0-9]*.h5'):
            id = re.split('[_.]', f)[1]
            models[id] = f
        assert len(models) == self.__chain_length
        for i in range(self.__chain_length):
            key = '%d' % (i+1)
            assert key in models
            self.__model[i].load(os.path.join(model_path, models[key]))
        self.__input_dim = self.__model[0].input_shape
        self.__output_dim = self.__model[0].output_shape

