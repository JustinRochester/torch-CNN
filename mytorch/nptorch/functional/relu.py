from ..GPU_np import np
from ..base.Tensor import Tensor


def relu(x: Tensor):
    ret = x.data
    ret[ret < 0] = 0
    if not x.requires_grad:
        return Tensor(ret)

    def grad_fn(grad):
        grad[ret == 0] = 0
        return grad
    return Tensor(
        data=ret,
        requires_grad=True,
        depend_on=[(x, grad_fn)]
    )
