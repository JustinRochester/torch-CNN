from ..GPU_np import np
from .Pool2D import Pool2D


def max_value(data):
    return np.max(data, axis=1)


def d_max_value(data):
    d = np.zeros_like(data)
    d[np.arange(data.shape[0]), np.argmax(data, axis=1)] = 1
    return d


class MaxPool2D(Pool2D):
    def __init__(self, input_size=(3, 32, 32), pooling_size=(2, 2)):
        super().__init__(input_size, pooling_size)
        self.f = max_value
        self.df = d_max_value
