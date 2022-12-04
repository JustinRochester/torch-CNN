import numpy as np
from img2col import convolution
from pool import max_pooling


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    y = np.copy(x)
    y[y <= 0] = 0
    return y


class CNN:
    def __init__(self, activation_function=relu):
        self.input1 = np.empty((1, 32, 32))
        self.filter1 = np.random.random_sample((6, 1, 5, 5)) * 0.1
        self.bias1 = np.random.random_sample((6, 1)) * 0.1
        self.output1 = np.empty((6, 28, 28))

        self.input2 = self.output1
        self.output2 = np.empty((6, 14, 14))

        self.input3 = self.output2
        self.filter3 = np.random.random_sample((16, 6, 5, 5)) * 0.01
        self.bias3 = np.random.random_sample((16, 1)) * 0.01
        self.output3 = np.empty((16, 10, 10))

        self.input4 = self.output3
        self.output4 = np.empty((16, 5, 5))

        self.input5 = self.output4
        self.filter5 = np.random.random_sample((120, 16, 5, 5)) * 0.01
        self.bias5 = np.random.random_sample((120, 1)) * 0.01
        self.output5 = np.empty((120, 1, 1))

        self.input6 = self.output5
        self.w6 = np.random.random_sample((84, 120)) * 0.01
        self.bias6 = np.random.random_sample((84, 1)) * 0.01
        self.output6 = np.empty((84, 1))

        self.input7 = self.output6
        self.w7 = np.random.random_sample((10, 84)) * 0.01
        self.bias7 = np.random.random_sample((10, 1)) * 0.01
        self.output7 = np.empty((10, 1))

        self.activation_function = activation_function

    def forward(self, image_array):
        x = np.copy(image_array)

        self.input1 = np.average(x, axis=0)
        x = convolution(image_array=x, filter_weight=self.filter1, filter_bias=self.bias1)
        x = self.activation_function(x)
        self.output1 = np.average(x, axis=0)

        x = max_pooling(x, (2, 2))
        self.output2 = np.average(x, axis=0)

        x = convolution(image_array=x, filter_weight=self.filter3, filter_bias=self.bias3)
        x = self.activation_function(x)
        self.output3 = np.average(x, axis=0)

        x = max_pooling(x, (2, 2))
        self.output4 = np.average(x, axis=0)

        x = convolution(image_array=x, filter_weight=self.filter5, filter_bias=self.bias5)
        x = self.activation_function(x)
        self.output5 = np.average(x, axis=0)

        x = x.reshape(x.shape[0], -1, 1)
        x = np.matmul(self.w6, x) + self.bias6
        x = self.activation_function(x)
        self.output6 = np.average(x, axis=0)

        x = np.matmul(self.w7, x) + self.bias7
        self.output7 = np.average(x, axis=0)

        return x


def softmax(x):
    x = np.exp(x - np.max(x, axis=1).reshape((-1, 1, 1)))
    return x / np.sum(x, axis=1).reshape((-1, 1, 1))


if __name__ == '__main__':
    nn = CNN()
    image_array = np.random.random_sample((100, 1, 32, 32))*100
    res = nn.forward(image_array)
    res = softmax(res).reshape(-1, 10).T
    print(res)
