from ..base import *
from .LossFunction import LossFunction
from ..functional import abs, sum, max


class Lp_Loss(LossFunction):
    """
    Calculate the loss function value with predict output and labels.
    It will calculate the L-p-distance between predict output and labels(p isn't infinite and p isn't 0).
    It calculates with the equation:
    L_p_loss = sum(|predict - labels| ** p) ** (1/p)
    """

    def __init__(self, p=1):
        super().__init__()
        self.p = p

    def __call__(self, *args, **kwargs):
        predict, labels = args
        loss = abs(predict - labels) ** self.p
        axis = tuple(range(1, len(loss.shape)))
        loss = sum(loss, axis=axis) ** (1 / self.p)
        return sum(loss)


class L_infinite_Loss(LossFunction):
    """
    Calculate the loss function value with predict output and labels.
    It will calculate the L-infinite-distance between predict output and labels.
    It calculates with the equation:
    L_infinite_loss = lim(p->infinite) [sum(|predict - labels| ** p) ** (1/p)]
                    = max( |predict - labels| )
    """

    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        predict, labels = args
        loss = abs(predict - labels)
        axis = tuple(range(1, len(loss.shape)))
        loss = max(loss, axis=axis)
        return sum(loss)


class L1_Loss(Lp_Loss):
    """
    Calculate the loss function value with predict output and labels.
    It will calculate the L-1-distance between predict output and labels.
    It calculates with the equation:
    L_1_loss = |predict - labels|
    """

    def __init__(self):
        super().__init__(p=1)


MAE_Loss = L1_Loss


class L2_Loss(Lp_Loss):
    """
    Calculate the loss function value with predict output and labels.
    It will calculate the L-2-distance between predict output and labels.
    It calculates with the equation:
    L_2_loss = sum( |predict - labels|**2 )**(1/2)
    """

    def __init__(self):
        super().__init__(p=2)


MSE_Loss = L2_Loss
