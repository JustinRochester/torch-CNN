from mytorch.nptorch.GPU_np import np
from ..base.Tensor import Tensor


def average(x: Tensor, axis=None, keepdims=False):
    if axis is None:
        axis = range(len(x.shape))
    if not isinstance(axis, tuple):
        axis = (axis,)
    ret = np.average(x.data, axis=axis, keepdims=keepdims)
    if not x.requires_grad:
        return Tensor(ret)

    grad_shape = list(x.shape)
    count_num = 1
    for dim in axis:
        count_num *= grad_shape[dim]
        grad_shape[dim] = 1
    grad_shape = tuple(grad_shape)

    def grad_fn(grad):
        return grad.reshape(grad_shape) * np.full(shape=x.shape, fill_value=1.0/count_num)

    return Tensor(
        data=ret,
        requires_grad=True,
        depend_on=[(x, grad_fn)]
    )


def mean(x: Tensor, axis=None, keepdim=False):
    return average(x, axis, keepdim)
