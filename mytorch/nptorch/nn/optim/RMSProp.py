from mytorch.nptorch.GPU_np import np
from ..base import *
from .Optimizer import Optimizer


eps = 1e-8


class RMSProp(Optimizer):
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

            second_momentum *= self.second_beta
            second_momentum += (1 - self.second_beta) * np.square(grad)

            pace = self.learning_rate * grad / np.sqrt(second_momentum + eps)
            data -= pace

    def get_data_list(self):
        return super().get_data_list() + self.second_momentum + [self.second_beta]

    def load_data_list(self, data_iter):
        super().load_data_list(data_iter)
        for momentum in self.second_momentum:
            momentum[:] = next(data_iter)
        self.second_beta[:] = next(data_iter)
