from ..base import *
from ..functional import sum, abs
from .LossFunction import LossFunction


class Regular_p_Loss(LossFunction):
    """
    Calculate the regular loss with power p(p isn't infinite).
    It accepts a list of parameters, calculate loss with sum(|parameter weight|**p)
    """
    def __init__(self, parameter_list=[], p=2):
        super().__init__()
        self.parameter_list = parameter_list
        self.p = p

    def __call__(self, *args, **kwargs):
        loss = Tensor(0)
        for parameter in self.parameter_list:
            loss += sum(abs(parameter) ** self.p)
        return sum(loss)


class Regular_1_Loss(Regular_p_Loss):
    def __init__(self, parameter_list=[]):
        super().__init__(parameter_list, 1)


class Regular_2_Loss(Regular_p_Loss):
    def __init__(self, parameter_list=[]):
        super().__init__(parameter_list, 2)
