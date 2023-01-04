from ..base.Tensor import Tensor


def relu(x: Tensor):
    ret = x.data
    mask = ret <= 0
    ret[mask] = 0
    if not x.requires_grad:
        return Tensor(ret)

    def grad_fn(grad):
        grad[mask] = 0
        return grad
    return Tensor(
        data=ret,
        requires_grad=True,
        depend_on=[(x, grad_fn)]
    )
