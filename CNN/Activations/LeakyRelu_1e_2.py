from ..GPU_np import np
from .Activation import Activation


class LeakyRelu_1e_2(Activation):
    @staticmethod
    def forward(x):
        y = np.copy(x)
        y[y < 0] *= 0.01
        return y

    @staticmethod
    def backward(x, y):
        d = np.copy(x)
        d[d > 0] = 1
        d[d <= 0] = 0.01
        return d
