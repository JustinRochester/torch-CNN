from ..base import *
from ..functional import log, sum
from mytorch.nptorch.GPU_np import np

eps = 1e-50


def softmax_cross_entropy(predict: Tensor, labels: Tensor):
    ret = predict.data
    ret -= np.max(ret, axis=1, keepdims=True)
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


class CrossEntropy:
    def __init__(self):
        pass

    def __call__(self, predict: Tensor, labels: Tensor, *args, **kwargs):
        cross_entropy = softmax_cross_entropy(predict, labels)
        return sum(cross_entropy)
