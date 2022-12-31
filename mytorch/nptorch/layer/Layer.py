import abc

from ..GPU_np import np
from ..base import *


class Layer:
    """
    Regulate the implement of all layers.
    """
    def __init__(self):
        self.parameter_list = []
        self.save_list = []

    @abc.abstractmethod
    def __call__(self, x: Tensor, *args, **kwargs):
        pass
