from mytorch.nptorch.GPU_np import np
from ..base import *
from .Layer import Layer
from ..functional import *


class ActivateFunctions(Layer):
    """
    Activate function layer.
    """

    def __init__(self, activate_function):
        super().__init__()
        self.activate_function = activate_function

    def forward(self, x: Tensor):
        return self.activate_function(x)


class ReLu(ActivateFunctions):
    def __init__(self):
        super().__init__(relu)


class Sigmoid(ActivateFunctions):
    def __init__(self):
        super().__init__(sigmoid)


class TanH(ActivateFunctions):
    def __init__(self):
        super().__init__(tanh)


activate_function_dict = {
    'relu': ReLu,
    'sigmoid': Sigmoid,
    'tanh': TanH,
}
