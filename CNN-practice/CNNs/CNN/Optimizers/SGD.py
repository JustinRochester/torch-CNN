from ..GPU_np import np
from .Optimizer import Optimizer


class SGDOptimizer(Optimizer):
    def __init__(self):
        super().__init__()

    def append(self, parameter, alpha=0):
        self.parameter_list.append(parameter)
        self.alpha_list.append(alpha)

    def update(self, learning_rate=1e-3):
        for i in range(len(self.parameter_list)):
            value, grad = self.parameter_list[i].value, self.parameter_list[i].grad
            grad += self.alpha_list[i] * value
            value -= learning_rate * grad
