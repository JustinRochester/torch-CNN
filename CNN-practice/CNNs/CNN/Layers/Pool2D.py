import abc

from ..GPU_np import np
from .Layer import Layer
from ..Base.img2col import img2col, col2img


class Pool2D(Layer):
    def __init__(self, input_size=(3, 32, 32), pooling_size=(2, 2)):
        super().__init__()
        self.input_size = input_size
        self.input_col = None
        self.f = None
        self.df = None
        self.pooling_y, self.pooling_x = pooling_size
        self.output_size = (input_size[0], input_size[1] // self.pooling_y, input_size[2] // self.pooling_x)

    def predict_forward(self, input_value):
        output = self.forward(input_value)
        self.input_col = None
        return output

    def forward(self, input_value):
        n, ic, ih, iw = input_value.shape
        oc, oh, ow = self.output_size
        self.input_col = img2col(
                            image_array=input_value,
                            filter_shape=(self.pooling_y, self.pooling_x),
                            stride=(self.pooling_y, self.pooling_x),
                            padding=0
                        ).reshape(n * oh * ow * ic, self.pooling_y * self.pooling_x)
        output = self.f(self.input_col)
        return output.reshape((n, oh, ow, ic)).transpose(0, 3, 1, 2)

    def backward(self, output_grad):
        n, oc, oh, ow = output_grad.shape
        output_grad = output_grad.transpose(0, 2, 3, 1).reshape((n * oc * oh * ow, 1))
        input_grad = self.df(self.input_col) * output_grad
        self.input_col = None
        return input_grad.reshape((n, ) + self.input_size)
