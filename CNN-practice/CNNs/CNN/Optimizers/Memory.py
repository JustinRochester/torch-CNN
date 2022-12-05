from ..GPU_np import np
from .Optimizer import Optimizer


class MemoryOptimizer(Optimizer):
    def __init__(self, rho=0.9):
        super().__init__()
        self.rho = rho

    def update(self, learning_rate=1e-3):
        for i in range(len(self.parameter_list)):
            value, grad = self.parameter_list[i].value, self.parameter_list[i].grad
            self.parameter_list[i].value = value * (1 - self.rho) + grad * self.rho
