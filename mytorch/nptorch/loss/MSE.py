from ..GPU_np import np
from ..base import *
from ..functional import square, sum


class MSE:
    def __init__(self):
        pass

    def __call__(self, predict: Tensor, labels: Tensor, *args, **kwargs):
        loss = square(predict - labels)
        return sum(loss)
