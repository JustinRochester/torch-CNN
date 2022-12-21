from ..GPU_np import np
from .Layer import Layer
from ..Base.Tensor import Tensor


class Linear(Layer):
    def __init__(self, input_size=10, output_size=10
                 ):
        super().__init__()
        self.input = None
        self.input_size = (input_size, 1)
        self.output_size = (output_size, 1)

        self.w = Tensor(
                                    shape=(input_size, output_size),
                                    initial_std=np.sqrt(2 / input_size)
                                )
        self.bias = Tensor(
                                       shape=(1, output_size),
                                       initial_std=np.sqrt(2 / input_size)
                                   )

        self.parameter_dict = {
            "w": self.w,
            "bias": self.bias,
        }

    def predict_forward(self, input_value):
        n, i, _ = input_value.shape
        output = input_value.reshape((n, -1)).dot(self.w.value)
        output = (output + self.bias.value).reshape((n, -1, 1))
        return output

    def forward(self, input_value):
        self.input = input_value
        return self.predict_forward(input_value)

    def backward(self, output_grad):
        n, o, _ = output_grad.shape
        i = self.input_size[0]
        output_grad = output_grad.reshape((n, o))
        self.bias.grad += np.sum(output_grad, axis=0).reshape((1, o))
        self.w.grad += self.input.reshape((n, -1)).T.dot(output_grad)
        input_grad = output_grad.dot(self.w.value.T)
        self.input = None
        return input_grad.reshape((n, i, 1))
