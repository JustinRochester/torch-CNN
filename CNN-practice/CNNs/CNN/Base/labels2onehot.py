from ..GPU_np import np


def labels2onehot(labels, class_number=None):
    """
    Change labels array(grouped by n*1) to the one hot vector array(grouped by n*class_number)
    """
    if class_number is None:
        class_number = np.max(labels)
    return (np.arange(class_number) == labels.reshape(-1)[:, None]) + 0


def probability2labels(probability):
    """
    Change probability array(from softmax or logistic, grouped by n*class_number) to the labels array(grouped by n*1)
    """
    n = probability.shape[0]
    return np.argmax(probability.reshape(n, -1), axis=1)
