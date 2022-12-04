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
                                    std=np.sqrt(2 / input_size)
                                )
        self.bias = NeuralVariable(
                                       shape=(output_size, 1),
                                       std=np.sqrt(2 / input_size)
                                   )

        self.parameter_dict = {
            "w": self.w,
            "bias": self.bias,
        }

    def predict_forward(self, input):
        n = input.shape[0]
        output = np.dot(self.w.value, input.reshape((n, -1)).T)
        output = (output + self.bias.value).T.reshape((n, -1, 1))
        return output

    def forward(self, input):
        self.input = input
        return self.predict_forward(input)

    def backward(self, output_grad):
        n = output_grad.shape[0]
        grad_output = output_grad
        self.bias.grad += np.sum(grad_output, axis=0)
        self.w.grad += np.sum(np.matmul(grad_output, self.input.transpose(0, 2, 1)), axis=0)
        input_grad = np.matmul(grad_output.reshape((n, -1)), self.w.value).reshape((n,) + self.input_size)
        self.input = None
        return input_grad
