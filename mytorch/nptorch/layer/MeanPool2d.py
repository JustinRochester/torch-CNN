from ..GPU_np import np
from ..base import *
from .Pool2d import Pool2d
from ..functional import mean


class MeanPool2d(Pool2d):
    def __init__(self, pooling_stride=(2, 2)):
        super().__init__(pooling_stride)

    def pooling(self, x: Tensor):
        return mean(x, axis=(3, 5))
