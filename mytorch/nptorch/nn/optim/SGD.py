from ..GPU_np import np
from ..base import *
from .Optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self,
                 parameter_list=[],
                 learning_rate=1e-3,
                 learning_rate_function=lambda x: x,
                 ):
        super().__init__(parameter_list, learning_rate, learning_rate_function)

    def step(self):
        super().step()
        for parameter in self.parameter_list:
            if not isinstance(parameter, Tensor):
                raise ValueError("Variables could not be updated.")
            data, grad = parameter.data, parameter.grad
            pace = self.learning_rate * grad
            data -= pace
