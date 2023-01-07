from mytorch.nptorch.GPU_np import np
from ..base import *
from .Optimizer import Optimizer


eps = 1e-8


class AdaGrad(Optimizer):
    def __init__(self,
                 parameter_list=[],
                 learning_rate=1e-3,
                 learning_rate_function=lambda x: x,
                 second_moment_beta=0.999,
                 ):
        super().__init__(parameter_list, learning_rate, learning_rate_function)
        self.second_beta = second_moment_beta

        self.second_momentum = []
        for e in self.parameter_list:
            self.second_momentum.append(np.zeros(e.shape))

    def step(self):
        super().step()
        for parameter, second_momentum in zip(self.parameter_list, self.second_momentum):
            if not isinstance(parameter, Tensor):
                raise ValueError("Variables could not be updated.")
            data, grad = parameter.data, parameter.grad

            second_momentum += np.square(grad)
            pace = self.learning_rate * grad / np.sqrt(second_momentum + eps)
            data -= pace
