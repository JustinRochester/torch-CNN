import numpy as np


def img2col(image_array, filter_size=(3, 3), stride=1):
    """
    :param image_array: shape = (number of images, channels of images, height of images, width of images)
    :param filter_size: (height of filters, width of filters)
    :param stride: integer
    :return: shape = (number of images, number of filters=1, number of pixels, channels * height * width)
    """
    n, c, h, w = image_array.shape
    fh, fw = filter_size
    h_size = (h - fh) // stride + 1
    w_size = (w - fw) // stride + 1
    weight_size = c * fh * fw
    convolution_array = np.empty((n, h_size * w_size, weight_size))
    for i in range(h_size):
        for j in range(w_size):
            convolution_array[:, i * w_size + j, :] = \
                image_array[:, :, i * stride: i * stride + fh, j * stride: j * stride + fw].reshape(n, -1)
    return convolution_array.reshape((n, 1, -1, weight_size))


def filter2row(filter_weight):
    """
    :param filter_weight: shape = (number of filters, channels of images, height of filters, width of filters)
    :return: shape = (number of images=1, number of filters, channels * height * width, 1)
    """
    o, i, h, w = filter_weight.shape
    return filter_weight.reshape((1, o, i * h * w, 1))


def bias2row(filter_bias):
    """
    :param filter_bias: shape = (number of filters, 1)
    :return: shape = (number of images=1, number of filters, number of pixels=1, 1)
    """
    o = filter_bias.shape[0]
    return filter_bias.reshape((1, o, 1, 1))


def convolution(image_array, filter_weight, filter_bias, stride=1):
    """
    :param image_array: shape = (number of images, channels of images, height of images, width of images)
    :param filter_weight: shape = (number of filters, channels of images, height of filters, width of filters)
    :param filter_bias: shape = (number of filters, 1)
    :param stride: integer
    :return: shape = (number of images, number of filters, new height of images, new width of images)
    """
    h, w = image_array.shape[2:]
    o, _, fh, fw = filter_weight.shape
    h_size = (h - fh) // stride + 1
    w_size = (w - fw) // stride + 1
    conv_array = np.matmul(img2col(image_array, (fh, fw), stride), filter2row(filter_weight))
    conv_array += bias2row(filter_bias)
    return conv_array.reshape((-1, o, h_size, w_size))


def col2img(column_array, filter_size=(3, 3), image_size=(3, 32, 32), stride=1):
    """
    :param column_array: shape = (number of images, number of pixels, channels * height * width)
    :param filter_size: (height of filters, width of filters)
    :param image_size: (channels of images, height of images, width of images)
    :param stride: integer
    :return: shape = (number of images, channels of images, height of images, width of images)
    """
    n, pixels, weight_size = column_array.shape
    c, h, w = image_size
    fh, fw = filter_size
    image_array = np.zeros((n, c, h, w))
    h_size = (h - fh) // stride + 1
    w_size = (w - fw) // stride + 1
    for i in range(h_size):
        for j in range(w_size):
            image_array[:, :, i * stride: i * stride + fh, j * stride: j * stride + fw] += \
                column_array[:, i * w_size + j, :].reshape(-1, c, fh, fw)
    return image_array


def row2filter(row_array, filter_size=(3, 32, 32)):
    """
    :param row_array: shape = (number of images, number of filters, channels * height * width, 1)
    :param filter_size: (channels of images, height of filters, width of filters)
    :return: shape = (number of filters, channels of images, height of filters, width of filters)
    """
    o = row_array.shape[1]
    i, h, w = filter_size
    filter_array = np.average(row_array, axis=0)
    return filter_array.reshape(o, i, h, w)


def row2bias(row_array):
    """
    :param row_array: shape = (number of images, number of filters, number of pixels, 1)
    :return: shape = (number of filters, 1)
    """
    return np.average(np.sum(row_array, axis=2), axis=0)


if __name__ == '__main__':
    img = np.random.random_sample((128, 3, 32, 32))
    filter_weight = np.random.random_sample((1024, 3, 5, 5))
    filter_bias = np.random.random_sample((1024, 1))
    x = img2col(img, filter_weight.shape[2:])
    print(x.shape)
    y = filter2row(filter_weight)
    print(y.shape)
    print(convolution(img, filter_weight, filter_bias).shape)

    x = col2img(column_array=x, filter_size=filter_weight.shape[2:], image_size=img.shape[1:])
    print(x.shape)
    x = row2filter(filter2row(filter_weight), filter_weight.shape[1:])
    print(x.shape)
    x = row2bias(bias2row(filter_bias))
    print(x.shape)
