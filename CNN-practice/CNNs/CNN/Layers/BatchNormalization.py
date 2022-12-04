from ..GPU_np import np
from .Layer import Layer
from ..Base.NeuralVariable import NeuralVariable


class BatchNormalization(Layer):
    def __init__(self, input_size=(1,), rho=0.9):
        super().__init__()
        self.input_size = input_size
        self.output_size = input_size
        self.channel_size = input_size[0]
        self.parameter_shape = (1, self.channel_size) + (1,) * len(input_size[1:])
        self.input_norm = None
        self.rho = rho

        self.mu = np.zeros(self.parameter_shape)
        self.sigma = np.zeros(self.parameter_shape)
        self.gamma = NeuralVariable(
            shape=self.parameter_shape,
            std=1
        )
        self.beta = NeuralVariable(
            shape=self.parameter_shape,
            std=1
        )
        self.gamma.value = np.ones(self.parameter_shape)
        self.beta.value = np.zeros(self.parameter_shape)
        self.parameter_dict = {
            "gamma": self.gamma,
            "beta": self.beta,
        }

    def predict_forward(self, input):
        input = (input - self.mu) / np.sqrt(self.sigma + 1e-9)
        output = input * self.gamma.value + self.beta.value
        return output

    def forward(self, input):
        n = input.shape[0]
        mu = np.average(input.reshape((n, self.channel_size, -1)), axis=2)
        mu = np.average(mu, axis=0).reshape(self.mu.shape)
        sigma = np.average(np.square(input - mu).reshape((n, self.channel_size, -1)), axis=2)
        sigma = np.average(sigma, axis=0).reshape(self.sigma.shape)

        self.mu = self.mu * (1 - self.rho) + mu * self.rho
        self.sigma = self.sigma * (1 - self.rho) + sigma * self.rho

        self.input_norm = (input - self.mu) / np.sqrt(self.sigma + 1e-9)
        output = self.input_norm * self.gamma.value + self.beta.value
        return output

    def backward(self, output_grad):
        n = output_grad.shape[0]
        input_grad = output_grad * self.gamma.value / np.sqrt(self.sigma + 1e-9)

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
        self.input_norm = None
        return input_grad
