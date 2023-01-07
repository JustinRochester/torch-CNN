from mytorch.nptorch.GPU_np import np
from ..base import *
from .Optimizer import Optimizer
from ..functional import sum, square, sqrt


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
        self.first_beta = Tensor(first_moment_beta)
        self.second_beta = Tensor(second_moment_beta)
        self.first_beta_pow = Tensor(1.0)
        self.second_beta_pow = Tensor(1.0)

        self.first_moment = []
        self.second_moment = []
        for e in self.parameter_list:
            self.first_moment.append(Tensor(np.zeros(e.shape)))
            self.second_moment.append(Tensor(np.zeros(e.shape)))

    def step(self):
        self.first_beta_pow *= self.first_beta
        self.second_beta_pow *= self.second_beta

        for i in range(len(self.parameter_list)):
            e = self.parameter_list[i]
            if not isinstance(e, Tensor):
                raise ValueError("Variables could not be updated.")
            data, grad = e.data, e.grad

            first_moment, second_moment = self.first_moment[i], self.second_moment[i]
            first_moment *= self.first_beta
            first_moment += (1 - self.first_beta) * grad
            second_moment *= self.second_beta
            second_moment += (1 - self.second_beta) * Tensor(np.square(grad))

            first_unbias = first_moment / (1 - self.first_beta_pow)
            second_unbias = second_moment / (1 - self.second_beta_pow)
            pace = self.learning_rate * first_unbias / sqrt(second_unbias + eps)
            e.data -= pace.data
