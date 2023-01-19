from ..base import *
from ..functional import log, sum
from ..GPU_np import np
from .LossFunction import LossFunction

eps = 1e-50


def softmax_cross_entropy(predict: Tensor, labels: Tensor):
    ret = predict.data
    ret = ret - np.max(ret, axis=1, keepdims=True)
    ret = np.exp(ret)
    ret /= np.sum(ret, axis=1, keepdims=True)

    if not predict.requires_grad:
        return Tensor(-np.log(ret + eps) * labels.data)

    def grad_fn(grad):
        return (ret - labels.data) * grad

    return Tensor(
        data=-np.log(ret + eps) * labels.data,
        requires_grad=True,
        depend_on=[(predict, grad_fn)]
    )


class CrossEntropySoftmax_Loss(LossFunction):
    """
    Calculate the loss function value with neural network's predict output and labels.
    It will transform predict output into softmax(predict).
    Then it will calculate cross entropy with softmax(predict) and labels by the equation:
    cross_entropy = -log( softmax(predict) )*labels.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        predict, labels = args
        cross_entropy = softmax_cross_entropy(predict, labels)
        return sum(cross_entropy)
