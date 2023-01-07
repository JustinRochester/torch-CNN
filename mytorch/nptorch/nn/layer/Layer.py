import abc

from ..base import *
from ..Module import Module


class Layer(Module):
    """
    Regulate the implement of all layers.
    """
    def __init__(self):
        super().__init__()

    def train_mode(self):
        super().train_mode()
        for parameter in self.parameter_list:
            parameter.requires_grad = True

    def predict_mode(self):
        super().predict_mode()
        for parameter in self.parameter_list:
            parameter.requires_grad = False

    @abc.abstractmethod
    def __call__(self, x: Tensor, *args, **kwargs):
        pass
