from ..GPU_np import np


def softmax(x):
    """
    Softmax x in the 2nd dimension, for all the data in the 1st dimension.
    Softmax the vector v = (v1, v2, ..., vm)^T and get u = (exp(v1), exp(v2), ... exp(vm))^T / sum(exp(vi))
    """
    n = x.shape[0]
    y = np.exp(x - np.max(x, axis=(1, 2)).reshape((n, 1, 1)))
    y /= np.sum(y, axis=(1, 2)).reshape((n, 1, 1))
    return y
