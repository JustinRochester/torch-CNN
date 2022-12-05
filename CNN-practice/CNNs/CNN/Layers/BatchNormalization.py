from ..GPU_np import np
from .Layer import Layer
from ..Base.NeuralVariable import NeuralVariable

eps = 1e-9


class BatchNormalization(Layer):
    def __init__(self, input_size=(1,), rho=0.9):
        super().__init__()
        self.input_size = input_size
        self.output_size = input_size
        self.channel_size = input_size[0]
        self.parameter_shape = (1, self.channel_size) + (1,) * len(input_size[1:])
        self.input_norm = None
        self.rho = rho

        self.count_batches = 0

        self.mu = np.zeros(self.parameter_shape)
        self.mu_update = np.zeros(self.parameter_shape)

        self.sigma2 = np.ones(self.parameter_shape)
        self.sigma2_update = np.zeros(self.parameter_shape)

        self.sigma = np.sqrt(self.sigma2 + eps)

        self.gamma = NeuralVariable(
            shape=self.parameter_shape,
            mu=1,
            std=0
        )
        self.beta = NeuralVariable(
            shape=self.parameter_shape,
            mu=0,
            std=0
        )
        self.parameter_dict = {
            "gamma": self.gamma,
            "beta": self.beta,
        }

    def update_statistics(self):
        if self.count_batches == 0:
            return
        self.mu = self.mu * (1 - self.rho) + self.mu_update / self.count_batches * self.rho
        self.sigma2 = self.sigma2 * (1 - self.rho) + self.sigma2_update / self.count_batches * self.rho
        self.mu_update = np.zeros_like(self.mu_update)
        self.sigma2_update = np.zeros_like(self.sigma2_update)
        self.sigma = np.sqrt(self.sigma2 + eps)
        self.count_batches = 0

    def predict_forward(self, input):
        input = (input - self.mu) / self.sigma
        output = input * self.gamma.value + self.beta.value
        return output

    def forward(self, input):
        n = input.shape[0]
        mu = np.average(input.reshape((n, self.channel_size, -1)), axis=2)
        mu = np.average(mu, axis=0).reshape(self.mu.shape)
        sigma2 = np.average(np.square(input - mu).reshape((n, self.channel_size, -1)), axis=2)
        sigma2 = np.average(sigma2, axis=0).reshape(self.sigma2.shape)

        self.count_batches += n
        self.mu_update += mu * n
        self.sigma2_update += sigma2 * n

        self.input_norm = (input - self.mu) / self.sigma
        output = self.input_norm * self.gamma.value + self.beta.value
        return output

    def backward(self, output_grad):
        n = output_grad.shape[0]

        norm_grad = (output_grad * self.gamma.value).reshape((n, self.channel_size, -1))
        self.input_norm = self.input_norm.reshape((n, self.channel_size, -1))
        output_grad = output_grad.reshape((n, self.channel_size, -1))
        self.gamma.grad += np.sum(
            np.average(output_grad * self.input_norm, axis=2),
            axis=0
        ).reshape(self.parameter_shape)
        self.beta.grad += np.sum(
            np.average(output_grad, axis=2),
            axis=0
        ).reshape(self.parameter_shape)

        sigma_grad = -np.average(norm_grad * self.input_norm, axis=2) / 2
        sigma_grad /= (self.sigma2.reshape((1, self.channel_size)) + eps)
        mu_grad = -np.average(norm_grad, axis=2) / self.sigma.reshape((1, self.channel_size)) \
                  - np.average(self.input_norm, axis=2) * sigma_grad * self.sigma.reshape((1, self.channel_size)) * 2 / n

        sigma_grad = sigma_grad.reshape((n,) + self.parameter_shape[1:])
        mu_grad = mu_grad.reshape((n,) + self.parameter_shape[1:])
        norm_grad = norm_grad.reshape((n,) + self.input_size)
        self.input_norm = self.input_norm.reshape((n,) + self.input_size)

        input_grad = norm_grad / self.sigma + (mu_grad + 2 * sigma_grad * self.sigma * self.input_norm) / n
        self.input_norm = None
        return input_grad
