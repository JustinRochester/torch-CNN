from ..GPU_np import np
from .Layer import Layer


class MeanPool2D(Layer):
    def __init__(self, input_size=(3, 32, 32), pooling_size=(2, 2)):
        super().__init__()
        self.input_size = input_size
        self.pooling_y, self.pooling_x = pooling_size
        self.output_size = (input_size[0], input_size[1] // self.pooling_y, input_size[2] // self.pooling_x)
        self.w = self.pooling_y * self.pooling_x

    def predict_forward(self, input):
        n = input.shape[0]
        c, h, w = self.output_size
        output = np.empty_like(input).reshape((n, c, h * w, -1))
        for y in range(h):
            for x in range(w):
                ly, ry = y * self.pooling_y, (y + 1) * self.pooling_y
                lx, rx = x * self.pooling_x, (x + 1) * self.pooling_x
                output[:, :, y * w + x, :] = input[:, :, lx: rx, ly: ry].reshape((n, c, 1, -1))
        output = np.average(output, axis=3).reshape((n, c, h, w))
        return output

    def forward(self, input):
        return self.predict_forward(input)

    def backward(self, output_grad):
        n = output_grad.shape[0]
        input_grad = np.zeros((n,) + self.input_size)
        for y in range(self.output_size[1]):
            for x in range(self.output_size[2]):
                ly, ry = y * self.pooling_y, (y+1) * self.pooling_y
                lx, rx = x * self.pooling_x, (x+1) * self.pooling_x
                input_grad[:, :, ly: ry, lx: rx] += output_grad[:, :, y: (y + 1), x: (x + 1)]
        input_grad /= self.w
        return input_grad
