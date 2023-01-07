from .base import *
from .interface import Savable


class Module(Savable):
    def __init__(self):
        self.parameter_list = []
        self.layer_list = []
        self.mode = 'train'

    def train_mode(self):
        if self.mode == 'train':
            return
        self.mode = 'train'
        for layer in self.layer_list:
            layer.train_mode()

    def predict_mode(self):
        if self.mode == 'predict':
            return
        self.mode = 'predict'
        for layer in self.layer_list:
            layer.predict_mode()

    def parameters(self):
        return self.parameter_list

    def __setattr__(self, key, value):
        self.__dict__[key] = value
        if type(value) is Parameter:
            self.parameter_list.append(value)
        elif isinstance(value, Module):
            self.parameter_list += value.parameter_list
            self.layer_list.append(value)

    def get_data_list(self):
        return [parameter.data for parameter in self.parameter_list]

    def load_data_list(self, data_iter):
        for parameter in self.parameter_list:
            parameter.data[:] = next(data_iter)
