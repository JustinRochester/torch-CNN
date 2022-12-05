from ..GPU_np import np
from .Layer import Layer
from ..Base.NeuralVariable import NeuralVariable
from ..Base.img2col import img2col, filter2row, bias2row, col2img, row2filter, row2bias


class Conv2D(Layer):
    def __init__(self, input_size=(3, 32, 32), filter_size=(3, 3), filter_num=10,
                 stride=1,
                 padding=0
                 ):
        super().__init__()
        self.input_size = input_size
        self.output_size = (
            filter_num,  # new channel
            (input_size[1] + 2 * padding - filter_size[0]) // stride + 1,  # new height
            (input_size[2] + 2 * padding - filter_size[1]) // stride + 1  # new width
        )
        self.filter_size = (filter_num, input_size[0]) + filter_size
        self.stride = stride
        self.padding = padding

        self.filter_array = NeuralVariable(
                                              shape=self.filter_size,
                                              std=np.sqrt(2 / (input_size[0] * filter_size[0] * filter_size[1]))
                                           )
        self.filter_bias = NeuralVariable(
                                              shape=(filter_num, 1),
                                              std=np.sqrt(2 / (input_size[0] * filter_size[0] * filter_size[1]))
                                           )
        self.input_col = None

        self.parameter_dict = {
            "filter_array": self.filter_array,
            "filter_bias": self.filter_bias,
        }

    def predict_forward(self, input):
        n = input.shape[0]
        input = img2col(input, self.filter_size[2:], self.stride, self.padding)
        output = np.dot(input, filter2row(self.filter_array.value))
        output += bias2row(self.filter_bias.value)
        output = output.reshape((n,) + self.output_size[1:] + (-1,)).transpose(0, 3, 1, 2)
        return output

    def forward(self, input):
        n = input.shape[0]
        self.input_col = img2col(input, self.filter_size[2:], self.stride, self.padding)
        output = np.dot(self.input_col, filter2row(self.filter_array.value))
        output += bias2row(self.filter_bias.value)
        output = output.reshape((n,) + self.output_size[1:] + (-1,)).transpose(0, 3, 1, 2)
        return output

    def backward(self, output_grad):
        n, o = output_grad.shape[:2]
        grad_output = output_grad.transpose(0, 2, 3, 1).reshape((-1, o))
        self.filter_bias.grad += row2bias(grad_output)
        self.filter_array.grad += row2filter(
            row_array=np.dot(self.input_col.T, grad_output),
            filter_size=self.filter_size
        )
        input_grad = col2img(
            column_array=np.dot(grad_output, filter2row(self.filter_array.value).T),
            image_size=(n,) + self.input_size,
            filter_size=self.filter_size[2:],
            stride=self.stride,
            padding=self.padding
        )
        self.input_col = None
        return input_grad
