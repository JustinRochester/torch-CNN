from ..GPU_np import np
from .Layer import Layer


class Flatten(Layer):
    def __init__(self, input_size=(10, 1, 1)):
        super().__init__()
        self.input_size = input_size
        self.output_size = (input_size[0] * input_size[1] * input_size[2], 1)

    def predict_forward(self, input_value):
        n = input_value.shape[0]
        return input_value.reshape((n, -1, 1))

    def forward(self, input_value):
        return self.predict_forward(input_value)

    def backward(self, output_grad):
        n = output_grad.shape[0]
        input_grad = output_grad.reshape((n,) + self.input_size)
        return input_grad
