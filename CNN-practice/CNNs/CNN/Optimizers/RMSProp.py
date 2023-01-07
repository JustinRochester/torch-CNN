from ..GPU_np import np
from .Optimizer import Optimizer


class RMSProp(Optimizer):
    def __init__(self, rho=0.9):
        super().__init__()
        self.grad_square = []
        self.rho = np.asarray([rho])

    def append(self, parameter, alpha=0):
        super().append(parameter, alpha)
        self.grad_square.append(np.zeros(parameter.shape))

    def update(self, learning_rate=1e-3):
        for i in range(len(self.parameter_list)):
            value, grad = self.parameter_list[i].value, self.parameter_list[i].grad
            grad += self.alpha_list[i] * value
            self.grad_square[i] = self.rho * self.grad_square[i] + (1 - self.rho) * np.square(grad)
            value -= learning_rate * grad / np.sqrt(self.grad_square[i] + 1e-8)

    def set_data(self, data_iter):
        super().set_data(data_iter)
        for i in range(len(self.grad_square)):
            self.grad_square[i] = next(data_iter)
        self.rho = next(data_iter)

    def get_data(self):
        return super().get_data() + self.grad_square + [self.rho]
