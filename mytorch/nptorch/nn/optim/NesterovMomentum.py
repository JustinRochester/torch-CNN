from mytorch.nptorch.GPU_np import np
from ..base import *
from .Optimizer import Optimizer


class NesterovMomentum(Optimizer):
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
        for parameter, first_moment in zip(self.parameter_list, self.first_momentum):
            if not isinstance(parameter, Tensor):
                raise ValueError("Variables could not be updated.")
            data, grad = parameter.data, parameter.grad

            new_momentum = self.first_beta * first_moment - self.learning_rate * grad
            data += new_momentum + self.first_beta * (new_momentum - first_moment)
            first_moment[:] = new_momentum

    def get_data_list(self):
        return super().get_data_list() + self.first_momentum + [self.first_beta]

    def load_data_list(self, data_iter):
        super().load_data_list(data_iter)
        for momentum in self.first_momentum:
            momentum[:] = next(data_iter)
        self.first_beta[:] = next(data_iter)
