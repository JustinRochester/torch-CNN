from ..GPU_np import np
from .Layer import Layer
from ..Activations import activation_dict


class Activation(Layer):
    def __init__(self, input_size=(1,), activation_name='relu'):
        super().__init__()
        self.input = None
        self.output = None
        self.input_size = self.output_size = input_size
        self.activation = activation_dict[activation_name]

    def predict_forward(self, input_value):
        return self.activation.forward(input_value)

    def forward(self, input_value):
        self.input = input_value
        self.output = self.predict_forward(input_value)
        return np.copy(self.output)

    def backward(self, output_grad):
        input_grad = self.activation.backward(self.input, self.output) * output_grad
        self.input = self.output = None
        return input_grad
