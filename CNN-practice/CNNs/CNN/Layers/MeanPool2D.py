from ..GPU_np import np
from .Pool2D import Pool2D


def mean_value(data):
    return np.mean(data, axis=1)


def d_mean_value(data):
    return np.full_like(data, 1.0 / data.shape[1])


class MeanPool2D(Pool2D):
    def __init__(self, input_size=(3, 32, 32), pooling_size=(2, 2)):
        super().__init__(input_size, pooling_size)
        self.f = mean_value
        self.df = d_mean_value
