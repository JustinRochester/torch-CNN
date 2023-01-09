from ..GPU_np import np
from ..base.Tensor import Tensor


def sigmoid(x: Tensor):
    ret = 1 / (1 + np.exp(-x.data))
    if not x.requires_grad:
        return Tensor(ret)

    def grad_fn(grad):
        return grad * ret * (1 - ret)
    return Tensor(
        data=ret,
        requires_grad=True,
        depend_on=[(x, grad_fn)]
    )
