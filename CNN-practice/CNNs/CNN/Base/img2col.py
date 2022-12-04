from ..GPU_np import np


def img2col(image_array, filter_size=(3, 3), stride=1, padding=0):
    """
    :param image_array: shape = (input_number, input_channels, input_height, input_width)
    :param filter_size: (filters_height, filters_width)
    :param stride: integer
    :param padding: integer
    :return: shape = (input_number * output_height * output_width, input_channels * filters_height * filters_width)
    """
    n, c, h, w = image_array.shape
    filter_h, filter_w = filter_size
    out_h = (h + 2 * padding - filter_h) // stride + 1
    out_w = (w + 2 * padding - filter_w) // stride + 1

    img = np.pad(image_array, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant')
    col = np.empty((n, c, filter_h, filter_w, out_h, out_w))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y: y_max: stride, x: x_max: stride]

    return col.transpose(0, 4, 5, 1, 2, 3).reshape((n * out_h * out_w, -1))


def filter2row(filter_weight):
    """
    :param filter_weight: shape = (filters_number, filters_channel, filters_height, filters_width)
    :return: shape = (filters_channel * filters_height * filters_width, filters_number)
    """
    o, i, h, w = filter_weight.shape
    return filter_weight.reshape((o, -1)).T


def bias2row(filter_bias):
    """
    :param filter_bias: shape = (filters_number, 1)
    :return: shape = (1, filters_number)
    """
    return filter_bias.T


def col2img(column_array, image_size=(3, 32, 32), filter_size=(3, 3), stride=1, padding=0):
    """
    :param column_array: shape = (input_number * output_height * output_width, input_channels * filters_height * filters_width)
    :param image_size: (input_number, input_channels, input_height, input_width)
    :param filter_size: (filters_height, filters_width)
    :param stride: integer
    :param padding: integer
    :return: shape = (number of images, channels of images, height of images, width of images)
    """
    n, c, h, w = image_size
    filter_h, filter_w = filter_size
    out_h = (h + 2 * padding - filter_h) // stride + 1
    out_w = (w + 2 * padding - filter_w) // stride + 1
    column_array = column_array.reshape((n, out_h, out_w, c, filter_h, filter_w)).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((n, c, h + 2 * padding + stride - 1, w + 2 * padding + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y: y_max: stride, x: x_max: stride] += column_array[:, :, y, x, :, :]
    return img[:, :, padding: padding + h, padding: padding + w]


def row2filter(row_array, filter_size=(1, 3, 32, 32)):
    """
    :param row_array: shape = (filters_channel * filters_height * filters_width, filters_number)
    :param filter_size: (filters_number, filters_channel, filters_height, filters_width)
    :return: shape = (filters_number, filters_channel, filters_height, filters_width)
    """
    return row_array.T.reshape(filter_size)


def row2bias(row_array):
    """
    :param row_array: shape = (input_number * output_height * output_width, filters_number)
    :return: shape = (number of filters, 1)
    """
    return np.sum(row_array, axis=0).reshape((-1, 1))


if __name__ == '__main__':
    img = np.random.random_sample((128, 3, 32, 32))
    filter_weight = np.random.random_sample((1024, 3, 5, 5))
    filter_bias = np.random.random_sample((1024, 1))
    stride = 1
    padding = 0
    out_h = (img.shape[2] + 2 * padding - filter_weight.shape[2]) // stride + 1
    out_w = (img.shape[3] + 2 * padding - filter_weight.shape[3]) // stride + 1

    x = img2col(img, filter_weight.shape[2:], stride, padding)
    print(x.shape)
    y = filter2row(filter_weight)
    print(y.shape)

    x = col2img(x, img.shape, filter_weight.shape[2:], stride, padding)
    print(x.shape)
    x = row2filter(filter2row(filter_weight), filter_weight.shape)
    print(x.shape)
    x = row2bias(bias2row(filter_bias))
    print(x.shape)

    z = np.matmul(img2col(img, filter_weight.shape[2:], stride, padding), filter2row(filter_weight))
    z += bias2row(filter_bias)
    print(z.shape)
    z = z.reshape((img.shape[0], out_h, out_w, filter_weight.shape[0])).transpose(0, 3, 1, 2)
    print(z.shape)
