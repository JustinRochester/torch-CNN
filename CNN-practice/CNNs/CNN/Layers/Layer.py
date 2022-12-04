import abc

import numpy as np


class Layer:
    def __init__(self):
        self.input_size = None
        self.output_size = None
        self.parameter_dict = {}

    @abc.abstractmethod
    def predict_forward(self, input):
        pass

    @abc.abstractmethod
    def forward(self, input):
        pass

    @abc.abstractmethod
    def backward(self, output_grad):
        pass

    def zero_grad(self):
        for name in self.parameter_dict.keys():
            parameter = getattr(self, name)
            parameter.grad = np.zeros_like(parameter.grad)

    def multi_grad(self, multiply=1):
        for name in self.parameter_dict.keys():
            parameter = getattr(self, name)
            parameter.grad *= multiply

    def build_model(self, optimizer):
        for name, value in self.parameter_dict.items():
            optimizer.append(value)

    def load_model(self, optimizer_iter):
        for name in self.parameter_dict.keys():
            value = next(optimizer_iter)
            setattr(self, name, value)
            self.parameter_dict[name] = value
