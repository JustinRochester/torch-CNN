from mytorch.nptorch.GPU_np import np
from ..base import *
from .Optimizer import Optimizer


class Momentum_SGD(Optimizer):
    def __init__(self,
                 parameter_list=[],
                 learning_rate=1e-3,
                 learning_rate_function=lambda x: x,
                 first_moment_beta=0.9,
                 ):
        super().__init__(parameter_list, learning_rate, learning_rate_function)
        self.first_beta = np.array([first_moment_beta])

        self.first_momentum = []
        for e in self.parameter_list:
            self.first_momentum.append(np.zeros(e.shape))

    def step(self):
        super().step()
        for parameter, first_momentum in zip(self.parameter_list, self.first_momentum):
            if not isinstance(parameter, Tensor):
                raise ValueError("Variables could not be updated.")
            data, grad = parameter.data, parameter.grad

            first_momentum *= self.first_beta
            first_momentum += grad

            pace = self.learning_rate * first_momentum
            data -= pace
