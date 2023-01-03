import abc

from ..GPU_np import np
from ..base import *
from ..Module import Module


class Layer(Module):
    """
    Regulate the implement of all layers.
    """
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def __call__(self, x: Tensor, *args, **kwargs):
        pass
