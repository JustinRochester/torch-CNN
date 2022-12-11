import numpy as np
from .Interfaces import *


class Sequential(Savable, Propagable):
    def __init__(self, *layer_list):
        self.layers = []
        self.add(*layer_list)

    def add(self, *layer_list):
        for layer in layer_list:
            self.layers.append(layer)

    def predict_forward(self, input_value):
        x = np.copy(input_value)
        for layer in self.layers:
            x = layer.predict_forward(x)
        return x

    def forward(self, input_value):
        x = np.copy(input_value)
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, output_grad):
        x = np.copy(output_grad)
        for layer in self.layers[::-1]:
            x = layer.backward(x)
        return x

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def multi_grad(self, multiply=1):
        for layer in self.layers:
            layer.multi_grad(multiply=multiply)

    def build_model(self, optimizer):
        for layer in self.layers:
            layer.build_model(optimizer)

    def get_data(self):
        lst = []
        for layer in self.layers:
            lst += layer.get_data()
        return lst

    def set_data(self, data_iter):
        for layer in self.layers:
            layer.set_data(data_iter)
