import numpy as np

from .Layers import *


class Sequential:
    def __init__(self, *layer_list):
        self.layers = []
        self.add(*layer_list)

    def add(self, *layer_list):
        for layer in layer_list:
            self.layers.append(layer)

    def predict_forward(self, input):
        x = np.copy(input)
        for layer in self.layers:
            x = layer.predict_forward(x)
        return x

    def forward(self, input):
        x = np.copy(input)
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, output_grad):
        x = np.copy(output_grad)
        for layer in self.layers[::-1]:
            x = layer.backward(x)
        return x

    def build_model(self, optimizer_dict):
        for layer in self.layers:
            layer.build_model(optimizer_dict)

    def load_model(self, optimizer_iter_dict):
        for layer in self.layers:
            layer.load_model(optimizer_iter_dict)
