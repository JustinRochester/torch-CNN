from mytorch.nptorch.GPU_np import np
from .LossFunction import LossFunction
from ..base import *


class LossCollector(LossFunction):
    """
    Collect a set of loss functions and their coefficients.
    It collects by a tuple like (loss functions: LossFunction, coefficient: Tensor)
    Each coefficient in the tuple, defaults by Tensor(1)
    """
    def __init__(self, *args):
        super().__init__()
        self.loss_list = []
        self.add(*args)

    def add(self, *args):
        for element in args:
            if not isinstance(element, tuple):
                try:
                    element = (element,)
                except Exception:
                    raise ValueError("Isn't a tuple of loss function and coefficient")

            if len(element) == 1:
                element = (*element, 1)
            elif len(element) > 2:
                raise ValueError("Isn't a tuple of loss function and coefficient")
            loss_function, coefficient = element

            if not isinstance(coefficient, Tensor):
                try:
                    coefficient = Tensor(coefficient)
                except Exception:
                    raise ValueError("Isn't a tuple of loss function and coefficient")
            if not isinstance(loss_function, LossFunction):
                raise ValueError("Isn't a tuple of loss function and coefficient")
            self.loss_list.append((loss_function, coefficient))

    def __call__(self, *args, **kwargs):
        loss = Tensor(0)
        for loss_function, coefficient in self.loss_list:
            loss += coefficient * loss_function(*args, **kwargs)
        return loss
