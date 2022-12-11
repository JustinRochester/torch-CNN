import abc
from ..GPU_np import np
from ..Interfaces.Savable import Savable


class Optimizer(Savable):
    def __init__(self):
        self.parameter_list = []
        self.alpha_list = []

    def append(self, parameter, alpha=0):
        self.parameter_list.append(parameter)
        self.alpha_list.append(np.asarray([alpha]))

    def regular_loss(self):
        loss_value = 0
        for i in range(len(self.parameter_list)):
            value = np.sum(np.square(self.parameter_list[i].value)) / 2
            loss_value += value * self.alpha_list[i]
        return loss_value

    @abc.abstractmethod
    def update(self, learning_rate=1e-3):
        pass

    def get_data(self):
        lst = []
        for parameter in self.parameter_list:
            lst += parameter.get_data()
        return lst + self.alpha_list

    def set_data(self, data_iter):
        for i in range(len(self.parameter_list)):
            self.parameter_list[i].set_data(data_iter)

        for i in range(len(self.alpha_list)):
            self.alpha_list[i] = next(data_iter)
