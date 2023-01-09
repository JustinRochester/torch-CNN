from ..GPU_np import np
from ..base.Tensor import Tensor


eps = 1e-8


def batch_normalization(x: Tensor, axis=None):
    if axis is None:
        axis = tuple(range(len(x.shape)))
    if not isinstance(axis, tuple):
        axis = (axis,)

    x_data = x.data
    sampling_mu = np.average(x_data, axis=axis, keepdims=True)
    sampling_sigma2 = np.average(np.square(x_data - sampling_mu), axis=axis, keepdims=True)
    sigma = np.sqrt(sampling_sigma2 + eps)
    x_std = (x_data - sampling_mu) / sigma
    if not x.requires_grad:
        return Tensor(x_std), Tensor(sampling_mu), Tensor(sampling_sigma2)

    def grad_fn(grad):
        m = x_std.shape[0]
        grad_sigma2 = -np.sum(grad * x_std, axis=axis, keepdims=True) / (2 * (sampling_sigma2 + eps))
        grad_mu = -np.sum(grad, axis=axis, keepdims=True) / sigma - \
                  2 * sigma * grad_sigma2 * np.sum(x_std, axis=axis, keepdims=True) / m
        return grad / sigma + (grad_mu + 2 * grad_sigma2 * x_std * sigma) / m

    return Tensor(
        data=x_std,
        requires_grad=True,
        depend_on=[(x, grad_fn)]
    ), Tensor(sampling_mu), Tensor(sampling_sigma2)
