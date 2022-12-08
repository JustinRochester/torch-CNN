from ..GPU_np import np
from .Layer import Layer
from ..Activations import activation_dict


class Dropout(Layer):
    def __init__(self, input_size=(1,), dropout_probability=0.5):
        super().__init__()
        self.input_size = self.output_size = input_size
        self.dropout_vector = np.empty(self.input_size)
        self.dropout_probability = dropout_probability

    def predict_forward(self, input_value):
        return input_value

    def forward(self, input_value):
        n = input_value.shape[0]
        self.dropout_vector = np.random.random_sample((n,) + self.input_size) < self.dropout_probability
        self.dropout_vector = self.dropout_vector + 0
        output = self.dropout_vector * input_value
        return output

    def backward(self, output_grad):
        input_grad = self.dropout_vector * output_grad / self.dropout_probability
        return input_grad
