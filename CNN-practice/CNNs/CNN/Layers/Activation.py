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

    def predict_forward(self, input):
        return self.activation.forward(input)

    def forward(self, input):
        self.input = input
        self.output = self.predict_forward(input)
        return np.copy(self.output)

    def backward(self, output_grad):
        input_grad = self.activation.backward(self.input, self.output) * output_grad
        self.input = self.output = None
        return input_grad
