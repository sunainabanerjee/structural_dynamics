import os
import logging
import numpy as np
from .utility import *
from .classifier import Classifier
from keras.layers.core import Dense
from keras.optimizers import RMSprop, Adam, SGD
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

__version__ = "1.0"
__all__ = ['MLP']


class MLP(Classifier):
    def __init__(self,
                 epochs=2500,
                 batch_size=10,
                 optimizer=Optimizers.adam(),
                 kernel_initializer=KernelInitializer.glorot_normal(),
                 loss=LossFunction.categorical_crossentropy()):
        assert optimizer in Optimizers.supported_list()
        assert kernel_initializer in KernelInitializer.supported_list()
        assert loss in LossFunction.supported_list()
        self.__name = "mlp"
        self.__model = None
        self.__input_dim = None
        self.__output_dim = None
        self.__train_history = None
        self.__layer_size = []
        self.__optimizer = optimizer
        self.__initializer = kernel_initializer
        self.__lr = 1e-3
        self.__metric = 'mae'
        self.__epochs = epochs
        self.__batch_size = batch_size
        self.__loss_function = loss
        self.__logger = logging.getLogger(os.path.basename(__file__)[:-3])

    @property
    def name(self):
        return self.__name

    @property
    def nepochs(self):
        return self.__epochs

    def set_epochs(self, epochs):
        if epochs > 0:
            self.__epochs = int(epochs)

    @property
    def optimizer(self):
        return self.__optimizer

    def set_optimizer(self, opt):
        assert opt in Optimizers.supported_list(filter='nn')
        self.__optimizer = opt

    @property
    def learning_rate(self):
        return self.__lr

    def set_learning_rate(self, lr):
        assert (lr > 0) and (lr <= 1.0)
        self.__lr = lr

    @property
    def input_shape(self):
        return self.__input_dim

    def set_input_shape(self, input_dim):
        assert (self.__input_dim is None) and (input_dim > 0)
        self.__input_dim = int(input_dim)
        self.__layer_size.insert(0, (self.__input_dim, None))

    @property
    def output_shape(self):
        return self.__output_dim

    def set_output_shape(self,
                         output_dim,
                         activation=Activations.softmax()):
        assert (self.__output_dim is None) and (output_dim > 0)
        assert activation in Activations.supported_list()
        self.__output_dim = int(output_dim)
        self.__layer_size.append((self.__output_dim,
                                  MLPLayerType.dense(),
                                  activation))

    @property
    def kernel_initializer(self):
        return self.__initializer

    def set_kernel_initializer(self, initializer):
        assert initializer in KernelInitializer.supported_list()
        self.__initializer = initializer

    @property
    def batch_size(self):
        return self.__batch_size

    def set_batch_size(self, size):
        if size > 0:
            self.__batch_size = int(size)

    @property
    def loss_function(self):
        return self.__loss_function

    def set_loss(self, loss_function):
        assert loss_function in LossFunction.supported_list()
        self.__loss_function = loss_function

    @property
    def nlayers(self):
        return len(self.__layer_size) - 1

    def add_layers(self,
                   dim,
                   layer_type=MLPLayerType.dense(),
                   activation=Activations.relu()):
        assert layer_type in MLPLayerType.supported_list()
        assert activation in Activations.supported_list()
        assert (self.__input_dim is not None) and (self.__output_dim is not None)
        assert (dim > 0) and (len(self.__layer_size) > 1)
        self.__layer_size.insert(-1, (int(dim), layer_type, activation))

    def set_model(self):
        assert self.input_shape is not None
        assert self.output_shape is not None
        assert self.__model is None
        assert (self.nlayers > 0)
        n = self.nlayers
        self.__model = Sequential()
        if self.__layer_size[1][1] == MLPLayerType.dense():
            self.__model.add(Dense(self.__layer_size[1][0],
                                   input_shape=(self.input_shape,),
                                   activation=self.__layer_size[1][2],
                                   kernel_initializer=self.kernel_initializer))
        for i in range(2, len(self.__layer_size)):
            if self.__layer_size[i][1] == MLPLayerType.dense():
                self.__model.add(Dense(self.__layer_size[i][0],
                                       activation=self.__layer_size[i][2],
                                       kernel_initializer=self.kernel_initializer))

        if self.optimizer == Optimizers.adam():
            opt = Adam(lr=self.learning_rate)
        elif self.optimizer == Optimizers.sgd():
            opt = SGD(lr=self.learning_rate)
        elif self.optimizer == Optimizers.rmsprop():
            opt = RMSprop(lr=self.learning_rate)
        self.__model.compile(loss=self.loss_function, optimizer=opt, metrics=[self.__metric])

    def summary(self):
        if self.__model is not None:
            self.__model.summary()

    def fit(self, x, y, test_split=0.2):
        assert len(self.__layer_size) > 1
        assert (test_split > 0) and (test_split <= 0.5)
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
        assert (len(x.shape) == 2) and (len(y.shape) <= 2)
        assert (x.shape[0] == y.shape[0])
        assert (x.shape[1] == self.input_shape)
        assert (y.shape[1] == self.output_shape)
        if self.__model is None:
            self.set_model()
        batch_size = self.__batch_size if x.shape[0] > self.__batch_size else x.shape[0]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split)
        self.__train_history = self.__model.fit(x_train, y_train, epochs=self.__epochs, batch_size=batch_size)
        accuracy = mean_absolute_error(y_test, self.predict(x_test))
        self.__logger.debug("Accuracy: %.2f" % accuracy)
        return accuracy

    def predict(self, x):
        assert self.__model is not None
        return self.__model.predict(x)

    def save(self, file_path):
        assert self.__model is not None
        assert os.path.isdir(os.path.dirname(file_path))
        assert file_path.endswith('.h5')
        self.__model.save(file_path)

    @property
    def has_training_history(self):
        return self.__train_history is not None

    @property
    def train_history(self):
        if self.has_training_history:
            return {'loss': self.__train_history.history['loss'],
                    'validation': self.__train_history.history['mean_absolute_error']}

    def load(self, model_path):
        assert os.path.isfile(model_path)
        self.__model = load_model(model_path)
        self.__input_dim = self.__model.input_shape[-1]
        self.__output_dim = self.__model.output_shape[-1]
        self.__layer_size = [(self.__input_dim, None)]
        for layer in self.__model.layers:
            sz = layer.output_shape[-1]
            layer_type = layer.get_config()['name'].split('_')[0]
            self.__layer_size.append((sz, layer_type))


