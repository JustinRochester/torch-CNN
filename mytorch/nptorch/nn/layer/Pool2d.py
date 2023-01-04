import abc

from ..base import *
from ..functional.im2col import im2col
from .Layer import Layer


class Pool2d(Layer):
    """
    Regulate the implement of pooling layers.
    """

    def __init__(self, pooling_stride=(2, 2)):
        super().__init__()
        self.pooling_stride = pooling_stride
        self.im2col = lambda x: im2col(x, filter_shape=pooling_stride, stride=pooling_stride, padding='none')

    @abc.abstractmethod
    def pooling(self, x: Tensor):
        """
        :param x: shape = (N, C, OH, FH, OW, FW)
        :return: shape = (N, C, OH, OW)
        """

    def __call__(self, x: Tensor, *args, **kwargs):
        n, c, ih, iw = x.shape
        ph, pw = self.pooling_stride
        oh, ow = ih // ph, iw // pw
        x = self.pooling(x.reshape((n, c, oh, ph, ow, pw)))
        return x
