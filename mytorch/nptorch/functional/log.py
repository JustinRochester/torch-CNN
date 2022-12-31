from ..GPU_np import np
from ..base.Tensor import Tensor


def log(x: Tensor):
    ret = np.log(x.data)
    if not x.requires_grad:
        return Tensor(ret)

    def grad_fn(grad):
        return grad / x
    return Tensor(
        data=ret,
        requires_grad=True,
        depend_on=[(x, grad_fn)]
    )
