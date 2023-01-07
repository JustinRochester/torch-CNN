from ..base import *
from ..functional import abs, sum


class MAE:
    def __init__(self):
        pass

    def __call__(self, predict: Tensor, labels: Tensor, *args, **kwargs):
        loss = abs(predict - labels)
        return sum(loss)
