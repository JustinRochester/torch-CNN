from ..GPU_np import np
from ..base.Tensor import Tensor


def square(x: Tensor):
    ret = np.square(x.data)
    if not x.requires_grad:
        return Tensor(ret)

    def grad_fn(grad):
        return grad * x.data * 2
    return Tensor(
        data=ret,
        requires_grad=True,
        depend_on=[(x, grad_fn)]
    )
