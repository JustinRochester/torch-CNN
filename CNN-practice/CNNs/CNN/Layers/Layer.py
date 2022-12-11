import abc
from ..GPU_np import np
from ..Interfaces import *
from ..Base.NeuralVariable import NeuralVariable


class Layer(Savable, Propagable):
    def __init__(self):
        self.input_size = None
        self.output_size = None
        self.parameter_dict = {}

    def set_data(self, data_iter):
        for name in self.parameter_dict.keys():
            parameter = self.parameter_dict[name]
            if not isinstance(parameter, NeuralVariable):
                raise TypeError("Not a NeuralVariable")
            parameter.set_data(data_iter)

    def get_data(self):
        lst = []
        for name in self.parameter_dict.keys():
            parameter = self.parameter_dict[name]
            if not isinstance(parameter, NeuralVariable):
                raise TypeError("Not a NeuralVariable")
            lst += parameter.get_data()
        return lst

    def zero_grad(self):
        for name in self.parameter_dict.keys():
            parameter = self.parameter_dict[name]
            if not isinstance(parameter, NeuralVariable):
                raise TypeError("Not a NeuralVariable")
            parameter.grad = np.zeros_like(parameter.grad)

    def multi_grad(self, multiply=1):
        for name in self.parameter_dict.keys():
            parameter = self.parameter_dict[name]
            if not isinstance(parameter, NeuralVariable):
                raise TypeError("Not a NeuralVariable")
            parameter.grad *= multiply

    def build_model(self, optimizer):
        for name in self.parameter_dict.keys():
            parameter = self.parameter_dict[name]
            if not isinstance(parameter, NeuralVariable):
                raise TypeError("Not a NeuralVariable")
            optimizer.append(parameter)
