from ..GPU_np import np
from .Layer import Layer
from ..Base.NeuralVariable import NeuralVariable


class Linear(Layer):
    def __init__(self, input_size=10, output_size=10
                 ):
        super().__init__()
        self.input = None
        self.input_size = (input_size, 1)
        self.output_size = (output_size, 1)

        self.w = NeuralVariable(
                                    shape=(output_size, input_size),
                                    initial_std=np.sqrt(2 / input_size)
                                )
        self.bias = NeuralVariable(
                                       shape=(output_size, 1),
                                       initial_std=np.sqrt(2 / input_size)
                                   )

        self.parameter_dict = {
            "w": self.w,
            "bias": self.bias,
        }

    def predict_forward(self, input_value):
        n = input_value.shape[0]
        output = self.w.value.dot(input_value.reshape((n, -1)).T)
        output = (output + self.bias.value).T.reshape((n, -1, 1))
        return output

    def forward(self, input_value):
        self.input = input_value
        return self.predict_forward(input_value)

    def backward(self, output_grad):
        n = output_grad.shape[0]
        grad_output = output_grad
        self.bias.grad += np.sum(grad_output, axis=0)
        h = grad_output.shape[1]
        w = self.input.shape[1]
        self.w.grad += grad_output.transpose(1, 0, 2).reshape((h, -1)).dot(
                                                self.input.transpose(0, 2, 1).reshape((-1, w))
                        )
        input_grad = grad_output.reshape((n, -1)).dot(self.w.value).reshape((n,) + self.input_size)
        self.input = None
        return input_grad
