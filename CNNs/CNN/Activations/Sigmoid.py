from ..GPU_np import np
from .Activation import Activation


class Sigmoid(Activation):
    @staticmethod
    def forward(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def backward(x, y):
        return y * (1 - y)
