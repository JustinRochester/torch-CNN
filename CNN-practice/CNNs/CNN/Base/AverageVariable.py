from ..GPU_np import np
from .NeuralData import NeuralData


class AverageVariable(NeuralData):
    def __init__(self, shape=(1,)):
        self.sum_value = np.zeros(shape)
        self.value = np.zeros(shape)
        self.count = np.asarray([0])

    def __add__(self, other):
        self.sum_value += other
        self.count += 1
        self.value = self.sum_value / self.count
        return self

    def load_data(self, data_iter):
        self.sum_value = next(data_iter)
        self.count = next(data_iter)
        self.value = self.sum_value / self.count

    def get_data(self):
        return [self.sum_value, self.count]
