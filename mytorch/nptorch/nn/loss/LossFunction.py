import abc

from ..base import *


class LossFunction:
    """
    Regulate the interfaces of loss functions.
    """
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass
