from mytorch.nptorch.GPU_np import np
from ..base import *
from .Optimizer import Optimizer


eps = 1e-8


class Adam(Optimizer):
    def __init__(self,
                 parameter_list=[],
                 learning_rate=1e-3,
                 learning_rate_function=lambda x: x,
                 first_moment_beta=0.9,
                 second_moment_beta=0.999,
                 ):
        super().__init__(parameter_list, learning_rate, learning_rate_function)
        self.first_beta = first_moment_beta
        self.second_beta = second_moment_beta
        self.first_beta_pow = np.array([1.0], dtype=np.float64)
        self.second_beta_pow = np.array([1.0], dtype=np.float64)

        self.first_momentum = []
        self.second_momentum = []
        for e in self.parameter_list:
            self.first_momentum.append(np.zeros(e.shape))
            self.second_momentum.append(np.zeros(e.shape))

    def step(self):
        super().step()
        self.first_beta_pow *= self.first_beta
        self.second_beta_pow *= self.second_beta

        for parameter, first_momentum, second_momentum in \
                zip(self.parameter_list, self.first_momentum, self.second_momentum):
            if not isinstance(parameter, Tensor):
                raise ValueError("Variables could not be updated.")
            data, grad = parameter.data, parameter.grad

            first_momentum *= self.first_beta
            first_momentum += (1 - self.first_beta) * grad
            second_momentum *= self.second_beta
            second_momentum += (1 - self.second_beta) * np.square(grad)

            first_unbias = first_momentum / (1 - self.first_beta_pow)
            second_unbias = second_momentum / (1 - self.second_beta_pow)
            pace = self.learning_rate * first_unbias / np.sqrt(second_unbias + eps)
            data -= pace

    def get_data_list(self):
        return super().get_data_list() +\
                self.first_momentum + \
                self.second_momentum + \
                [self.first_beta, self.second_beta] + \
                [self.first_beta_pow, self.second_beta_pow]

    def load_data_list(self, data_iter):
        super().load_data_list(data_iter)
        for momentum in self.first_momentum:
            momentum[:] = next(data_iter)
        for momentum in self.second_momentum:
            momentum[:] = next(data_iter)
        self.first_beta[:] = next(data_iter)
        self.second_beta[:] = next(data_iter)
        self.first_beta_pow[:] = next(data_iter)
        self.second_beta_pow[:] = next(data_iter)
