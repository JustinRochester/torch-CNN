from ..GPU_np import np
from .Optimizer import Optimizer


class AdamOptimizer(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999):
        super().__init__()
        self.first_moment = []
        self.second_moment = []
        self.beta1 = np.asarray([beta1])
        self.beta2 = np.asarray([beta2])
        self.pow_beta1 = np.asarray([1.0])
        self.pow_beta2 = np.asarray([1.0])

    def append(self, parameter, alpha=0):
        super().append(parameter, alpha)
        self.first_moment.append(np.zeros(parameter.shape))
        self.second_moment.append(np.asarray([0.0]))

    def update(self, learning_rate=1e-3):
        self.pow_beta1 *= self.beta1
        self.pow_beta2 *= self.beta2
        for i in range(len(self.parameter_list)):
            value, grad = self.parameter_list[i].value, self.parameter_list[i].grad
            grad += self.alpha_list[i] * value
            self.first_moment[i] = self.beta1 * self.first_moment[i] + (1 - self.beta1) * grad
            self.second_moment[i] = self.beta2 * self.second_moment[i] + (1 - self.beta2) * np.sum(np.square(grad))
            first_unbias = self.first_moment[i] / (1 - self.pow_beta1)
            second_unbias = self.second_moment[i] / (1 - self.pow_beta2)
            value -= learning_rate * first_unbias / (np.sqrt(second_unbias) + 1e-8)

    def set_data(self, data_iter):
        super().set_data(data_iter)
        for i in range(len(self.first_moment)):
            self.first_moment[i] = next(data_iter)

        for i in range(len(self.second_moment)):
            self.second_moment[i] = next(data_iter)

        self.beta1 = next(data_iter)
        self.pow_beta1 = next(data_iter)
        self.beta2 = next(data_iter)
        self.pow_beta2 = next(data_iter)

    def get_data(self):
        return super().get_data()\
               + self.first_moment\
               + self.second_moment\
               + [self.beta1, self.pow_beta1, self.beta2, self.pow_beta2]

