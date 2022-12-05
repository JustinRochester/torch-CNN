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

    def build_model(self, optimizer_dict):
        for parameter, optimizer_name in self.parameter_dict.items():
            optimizer_dict[optimizer_name].append(
                                                    getattr(self, parameter)
                                                 )

    def load_model(self, optimizer_iter_dict):
        for parameter, optimizer_name in self.parameter_dict.items():
            setattr(
                        self,
                        parameter,
                        next(optimizer_iter_dict[optimizer_name])
                    )
