from ..GPU_np import np
from .Optimizer import Optimizer


class SGD_with_Momentum(Optimizer):
    def __init__(self, rho=0.9):
        super().__init__()
        self.momentum = []
        self.rho = np.asarray([rho])

    def append(self, parameter, alpha=0):
        super().append(parameter, alpha)
        self.momentum.append(np.zeros(parameter.shape))

    def update(self, learning_rate=1e-3):
        for i in range(len(self.parameter_list)):
            value, grad = self.parameter_list[i].value, self.parameter_list[i].grad
            grad += self.alpha_list[i] * value
            self.momentum[i] = self.momentum[i] * self.rho + grad
            value -= learning_rate * self.momentum[i]

    def set_data(self, data_iter):
        super().set_data(data_iter)
        for i in range(len(self.momentum)):
            self.momentum[i] = next(data_iter)
        self.rho = next(data_iter)

    def get_data(self):
        return super().get_data() + self.momentum + [self.rho]
