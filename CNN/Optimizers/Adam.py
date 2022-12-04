from ..GPU_np import np
from .Optimizer import Optimizer


class AdamOptimizer(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999):
        super().__init__()
        self.first_moment = []
        self.second_moment = []
        self.beta1 = beta1
        self.beta2 = beta2
        self.pow_beta1 = 1
        self.pow_beta2 = 1
        self.load_list = [self.first_moment, self.second_moment]

    def append(self, parameter, alpha=0):
        self.parameter_list.append(parameter)
        self.alpha_list.append(alpha)
        self.first_moment.append(np.zeros(parameter.shape))
        self.second_moment.append(np.zeros((1,)))

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
