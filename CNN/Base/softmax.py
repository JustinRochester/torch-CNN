from ..GPU_np import np


def softmax(x):
    n = x.shape[0]
    y = np.exp(x - np.max(x, axis=(1, 2)).reshape((n, 1, 1)))
    y /= np.sum(y, axis=(1, 2)).reshape((n, 1, 1))
    return y