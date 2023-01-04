from mytorch.nptorch.GPU_np import np
from ..base.Tensor import Tensor


def exp(x: Tensor):
    ret = np.exp(x.data)
    if not x.requires_grad:
        return Tensor(ret)

    def grad_fn(grad):
        return grad * ret
    return Tensor(
        data=ret,
        requires_grad=True,
        depend_on=[(x, grad_fn)]
    )
