from ..GPU_np import np
from ..base.Tensor import Tensor


def sum(x: Tensor, axis=None, keepdims=False):
    if axis is None:
        axis = range(len(x.shape))
    if not isinstance(axis, tuple):
        axis = (axis,)
    ret = np.sum(x.data, axis=axis, keepdims=keepdims)
    if not x.requires_grad:
        return Tensor(ret)

    grad_shape = list(x.shape)
    for dim in axis:
        grad_shape[dim] = 1
    grad_shape = tuple(grad_shape)

    def grad_fn(grad):
        return grad.reshape(grad_shape) * np.ones(x.shape)

    return Tensor(
        data=ret,
        requires_grad=True,
        depend_on=[(x, grad_fn)]
    )
