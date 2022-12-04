from ..GPU_np import np
from .Activation import Activation


class Relu(Activation):
    @staticmethod
    def forward(x):
        y = np.copy(x)
        y[y < 0] = 0
        return y

    @staticmethod
    def backward(x, y):
        d = np.copy(y)
        d[d > 0] = 1
        return d
