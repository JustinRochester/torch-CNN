import abc
from ..GPU_np import np


class Optimizer:
    def __init__(self):
        self.parameter_list = []
        self.alpha_list = []
        self.load_list = []

    def zero_grad(self):
        for i in range(len(self.parameter_list)):
            self.parameter_list[i].grad = np.zeros_like(self.parameter_list[i].grad)

    def multi_grad(self, multiply=1):
        for i in range(len(self.parameter_list)):
            self.parameter_list[i].grad *= multiply

    def append(self, parameter, alpha=0):
        self.parameter_list.append(parameter)
        self.alpha_list.append(alpha)

    def regular_loss(self):
        loss_value = 0
        for i in range(len(self.parameter_list)):
            value = np.sum(np.square(self.parameter_list[i].value)) / 2
            loss_value += value * self.alpha_list[i]
        return loss_value

    @abc.abstractmethod
    def update(self, learning_rate=1e-3):
        pass

    def get_iter(self):
        return iter(self.parameter_list)

    def load_data(self):
        for lst in self.load_list:
            for i in range(len(lst)):
                lst[i] = np.asarray(lst[i].tolist())
        for i in range(len(self.parameter_list)):
            self.parameter_list[i].value = np.asarray(self.parameter_list[i].value.tolist())
            self.parameter_list[i].grad = np.asarray(self.parameter_list[i].grad.tolist())
