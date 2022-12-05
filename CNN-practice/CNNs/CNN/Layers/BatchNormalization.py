from ..GPU_np import np
from .Layer import Layer
from ..Base.NeuralVariable import NeuralVariable


class BatchNormalization(Layer):
    def __init__(self,
                 input_size=(1,),
                 gamma_optimizer='Adam',
                 beta_optimizer='Adam',
                 mu_optimizer='Memory',
                 sigma_optimizer='Memory',
                 ):
        super().__init__()
        self.input_size = input_size
        self.output_size = input_size
        self.channel_size = input_size[0]
        self.parameter_shape = (1, self.channel_size) + (1,) * len(input_size[1:])
        self.input_norm = None

        self.mu = NeuralVariable(shape=self.parameter_shape, mu=0, std=0)
        self.sigma = NeuralVariable(shape=self.parameter_shape, mu=1, std=0)
        self.gamma = NeuralVariable(shape=self.parameter_shape, mu=1, std=0)
        self.beta = NeuralVariable(shape=self.parameter_shape, mu=0, std=0)
        self.parameter_dict = {
            "gamma": gamma_optimizer,
            "beta": beta_optimizer,
            "mu": mu_optimizer,
            "sigma": sigma_optimizer
        }

    def predict_forward(self, input):
        input = (input - self.mu.value) / np.sqrt(self.sigma.value + 1e-9)
        output = input * self.gamma.value + self.beta.value
        return output

    def forward(self, input):
        n = input.shape[0]
        mu = np.average(input.reshape((n, self.channel_size, -1)), axis=2)
        mu = np.average(mu, axis=0).reshape(self.mu.shape)
        sigma = np.average(np.square(input - mu).reshape((n, self.channel_size, -1)), axis=2)
        sigma = np.average(sigma, axis=0).reshape(self.sigma.shape)

        self.mu.grad += mu * n
        self.sigma.grad += sigma * n

        self.input_norm = (input - self.mu.value) / np.sqrt(self.sigma.value + 1e-9)
        output = self.input_norm * self.gamma.value + self.beta.value
        return output

    def backward(self, output_grad):
        n = output_grad.shape[0]
        input_grad = output_grad * self.gamma.value / np.sqrt(self.sigma.value + 1e-9)

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
