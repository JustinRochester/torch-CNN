from ..GPU_np import np


class NeuralVariable:
    def __init__(self, shape=(1,), mu=0, std=1):
        self.shape = shape
        self.value = np.random.normal(mu, std, shape)
        self.grad = np.zeros(shape)
