from ..GPU_np import np
from ..base import *
from ..functional import log, sum


eps = 1e-50


class CrossEntropy:
    def __init__(self):
        pass

    def __call__(self, predict: Tensor, labels: Tensor, *args, **kwargs):
        cross_entropy = -log(predict * labels + eps)
        return sum(cross_entropy)
