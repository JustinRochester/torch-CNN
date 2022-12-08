from ..GPU_np import np
from .Layer import Layer
from ..Base.NeuralVariable import NeuralVariable
from ..Base.AverageVariable import AverageVariable

eps = 1e-9


class BatchNormalization(Layer):
    def __init__(self, input_size=(1,)):
        super().__init__()
        self.input_size = input_size
        self.output_size = input_size
        self.channel_size = input_size[0]
        self.parameter_shape = (1, self.channel_size) + (1,) * len(input_size[1:])
        self.input_norm = None

        self.batch_mu = np.zeros(self.parameter_shape)
        self.mu = AverageVariable(self.parameter_shape)

        self.batch_sigma2 = np.ones(self.parameter_shape)
        self.sigma2 = AverageVariable(self.parameter_shape)

        self.batch_sigma = np.sqrt(self.batch_sigma2 + eps)

        self.slope = np.ones(self.parameter_shape)
        self.bias = np.zeros(self.parameter_shape)

        self.gamma = NeuralVariable(
            shape=self.parameter_shape,
            initial_mu=1,
            initial_std=0
        )
        self.beta = NeuralVariable(
            shape=self.parameter_shape,
            initial_mu=0,
            initial_std=0
        )
        self.parameter_dict = {
            "gamma": self.gamma,
            "beta": self.beta,
        }

    def predict_forward(self, input_value):
        return self.slope * input_value + self.bias

    def forward(self, input_value):
        n = input_value.shape[0]
        mu = np.average(input_value.reshape((n, self.channel_size, -1)), axis=2)
        mu = np.average(mu, axis=0).reshape(self.batch_mu.shape)
        sigma2 = np.average(np.square(input_value - mu).reshape((n, self.channel_size, -1)), axis=2)
        sigma2 = np.average(sigma2, axis=0).reshape(self.batch_sigma2.shape)

        self.batch_mu = mu
        self.batch_sigma2 = sigma2
        self.batch_sigma = np.sqrt(self.batch_sigma2 + eps)

        self.mu += self.batch_mu
        self.sigma2 += self.batch_sigma2

        self.input_norm = (input_value - self.batch_mu) / self.batch_sigma
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
        sigma_grad /= (self.batch_sigma2.reshape((1, self.channel_size)) + eps)
        mu_grad = -np.average(norm_grad, axis=2) / self.batch_sigma.reshape((1, self.channel_size)) \
                  - np.average(self.input_norm, axis=2) * sigma_grad * self.batch_sigma.reshape((1, self.channel_size)) * 2 / n

        sigma_grad = sigma_grad.reshape((n,) + self.parameter_shape[1:])
        mu_grad = mu_grad.reshape((n,) + self.parameter_shape[1:])
        norm_grad = norm_grad.reshape((n,) + self.input_size)
        self.input_norm = self.input_norm.reshape((n,) + self.input_size)

        input_grad = norm_grad / self.batch_sigma + (mu_grad + 2 * sigma_grad * self.batch_sigma * self.input_norm) / n
        self.input_norm = None

        sigma = np.sqrt(self.sigma2.value + eps)
        self.slope = self.gamma.value / sigma
        self.bias = self.beta.value - self.gamma.value * self.mu.value / sigma
        return input_grad

    def get_data(self):
        return self.mu.get_data() + self.sigma2.get_data()

    def load_data(self, data_iter):
        self.mu.load_data(data_iter)
        self.sigma2.load_data(data_iter)
        sigma = np.sqrt(self.sigma2.value + eps)
        self.slope = self.gamma.value / sigma
        self.bias = self.beta.value - self.gamma.value * self.mu.value / sigma
