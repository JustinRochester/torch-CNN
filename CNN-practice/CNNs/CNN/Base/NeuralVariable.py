from ..GPU_np import np


class NeuralVariable:
    def __init__(self, shape=(1,), std=1):
        self.shape = shape
        self.id = -1
        self.value = np.random.normal(0, std, shape)
        self.grad = np.empty(shape)
