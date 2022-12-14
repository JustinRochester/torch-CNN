import abc
from ..interface import Savable


class Optimizer(Savable):
    def __init__(self,
                 parameter_list=[],
                 learning_rate=1e-3,
                 learning_rate_function=lambda x: x,
                 ):
        self.parameter_list = parameter_list
        self.learning_rate = learning_rate
        self.learning_rate_function = learning_rate_function

    @abc.abstractmethod
    def step(self):
        self.learning_rate = self.learning_rate_function(self.learning_rate)

    def get_data_list(self):
        return []

    def load_data_list(self, data_iter):
        pass
