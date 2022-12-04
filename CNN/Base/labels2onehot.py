from ..GPU_np import np


def labels2onehot(labels, class_number=None):
    if class_number is None:
        class_number = np.max(labels)
    return (np.arange(class_number) == labels.reshape(-1)[:, None]) + 0


def probability2labels(probability):
    n = probability.shape[0]
    return np.argmax(probability.reshape(n, -1), axis=1)
