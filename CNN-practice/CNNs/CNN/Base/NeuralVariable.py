from ..GPU_np import np
from ..Interfaces.Savable import Savable


class NeuralVariable(Savable):
    def __init__(self, shape=(1,), initial_mu=0, initial_std=1):
        self.shape = shape
        self.value = np.random.normal(initial_mu, initial_std, shape)
        self.grad = np.empty(shape)

    def set_data(self, data_iter):
        self.value = next(data_iter)
        self.grad = next(data_iter)

    def get_data(self):
        return [self.value, self.grad]
