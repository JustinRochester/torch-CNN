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
        self.parameter_shape = (1, self.channel_size, 1)
        self.input_norm = None
        self.rho = rho

        self.sample_mu = np.zeros(self.parameter_shape)
        self.sample_sigma2 = np.ones(self.parameter_shape)
        self.sample_sigma = np.sqrt(self.sample_sigma2 + eps)

        self.running_mu = np.zeros(self.parameter_shape)
        self.running_sigma2 = np.ones(self.parameter_shape)
        self.running_sigma = np.sqrt(self.running_sigma2 + eps)

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

    def predict_forward(self, input):
        n, c = input.shape[:2]
        input = (input.reshape(n, c, -1) - self.running_mu) / self.running_sigma
        output = input * self.gamma.value + self.beta.value
        return output.reshape((n,) + self.output_size)

    def forward(self, input):
        n, c = input.shape[:2]
        input = input.reshape((n, c, -1))
        average_size = n * input.shape[2]

        self.sample_mu = np.einsum('ijk->j', input) / average_size
        self.sample_mu = self.sample_mu.reshape(self.parameter_shape)
        self.sample_sigma2 = np.einsum('ijk->j', np.square(input-self.sample_mu)) / average_size
        self.sample_sigma2 = self.sample_sigma2.reshape(self.parameter_shape)
        self.sample_sigma = np.sqrt(self.sample_sigma2 + eps)

        self.running_mu = self.running_mu * (1 - self.rho) + self.sample_mu * self.rho
        self.running_sigma2 = self.running_sigma2 * (1 - self.rho) + self.sample_sigma2 * self.rho
        self.running_sigma = np.sqrt(self.running_sigma2 + eps)

        self.input_norm = (input - self.sample_mu) / self.sample_sigma
        output = self.input_norm * self.gamma.value + self.beta.value
        return output.reshape((n,) + self.output_size)

    def backward(self, output_grad):
        n, c = output_grad.shape[:2]
        output_grad = output_grad.reshape((n, c, -1))

        gamma_grad = np.einsum('ijk->j', output_grad * self.input_norm) / output_grad.shape[2]
        self.gamma.grad += gamma_grad.reshape(self.parameter_shape)

        beta_grad = np.einsum('ijk->j', output_grad) / output_grad.shape[2]
        self.beta.grad += beta_grad.reshape(self.parameter_shape)

        norm_grad = output_grad * self.gamma.value
        sigma2_grad = np.einsum('ijk->j', norm_grad * self.input_norm) / output_grad.shape[2]
        sigma2_grad = -sigma2_grad.reshape(self.parameter_shape) / (2 * (self.sample_sigma2 + eps))
        mu_grad = norm_grad / self.sample_sigma + 2 * self.sample_sigma * sigma2_grad * self.input_norm / n
        mu_grad = - np.einsum('ijk->j', mu_grad).reshape(self.parameter_shape) / output_grad.shape[2]

        input_grad = norm_grad / self.sample_sigma + (mu_grad + 2 * sigma2_grad * self.sample_sigma * self.input_norm) / n
        self.input_norm = None
        return input_grad.reshape((n,) + self.output_size)
