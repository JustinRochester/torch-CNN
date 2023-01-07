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

        self.first_moment = []
        for e in self.parameter_list:
            self.first_moment.append(np.zeros(e.shape))

    def step(self):
        super().step()
        for parameter, first_moment in zip(self.parameter_list, self.first_moment):
            if not isinstance(parameter, Tensor):
                raise ValueError("Variables could not be updated.")
            data, grad = parameter.data, parameter.grad

            new_momentum = self.first_beta * first_moment - self.learning_rate * grad
            data += new_momentum + self.first_beta * (new_momentum - first_moment)
            first_moment[:] = new_momentum
