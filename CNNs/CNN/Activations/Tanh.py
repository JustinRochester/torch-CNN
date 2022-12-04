from ..GPU_np import np
from .Activation import Activation


class Tanh(Activation):
    @staticmethod
    def forward(x):
        return np.tanh(x)

    @staticmethod
    def backward(x, y):
        return y * np.atanh(x)
