from ..GPU_np import np
from ..base.Tensor import Tensor


def max(x: Tensor, axis=None, keepdims=False):
    if axis is None:
        axis = range(len(x.shape))
    if not isinstance(axis, tuple):
        axis = (axis,)
    ret = np.max(x.data, axis=axis, keepdims=keepdims)
    if not x.requires_grad:
        return Tensor(ret)

    grad_shape = list(x.shape)
    mask = [1 for i in range(len(x.shape))]
    for dim in axis:
        mask[dim] = grad_shape[dim]
        grad_shape[dim] = 1
    grad_shape = tuple(grad_shape)

    mask = np.tile(ret.reshape(grad_shape), mask) != x.data

    def grad_fn(grad):
        return grad.reshape(grad_shape) * mask

    return Tensor(
        data=ret,
        requires_grad=True,
        depend_on=[(x, grad_fn)]
    )
