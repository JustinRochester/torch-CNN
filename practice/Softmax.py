import numpy as np


def softmax(v):
    u = np.exp(v - np.max(v))
    index = np.argmax(u)
    val = u[index] / np.sum(u)
    return index, val


if __name__ == '__main__':
    test = np.random.rand(6)
    print(test)
    print(softmax(test))
