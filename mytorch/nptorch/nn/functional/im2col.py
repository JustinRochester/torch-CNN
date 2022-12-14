from ..GPU_np import np
from ..base import *


def get_stride_dim(stride):
    """
    :param stride: integer means the same strides in all dimension,
                   tuple means the different strides in each dimension
    :return: stride in h-dimension, stride in w-dimension
    """
    if isinstance(stride, tuple):
        if len(stride) == 2:
            stride_h, stride_w = stride
        else:
            raise TypeError("Unacceptable stride type")
    else:
        stride_h = stride_w = stride
    if not isinstance(stride_h, int) or not isinstance(stride_w, int):
        raise TypeError("Unacceptable stride type")
    return stride_h, stride_w


def get_padding_dim(padding, filter_h, filter_w):
    """
    :param padding: integer means the same paddings in all dimension,
                    tuple means the different paddings in each dimension,
                    string 'same' means keep the image shape,
                    string 'none' means 0 in all dimension
    :param filter_h: filter's height
    :param filter_w: filter's width
    :return: padding in h-dimension, padding in w-dimension
    """
    if isinstance(padding, tuple):
        if len(padding) == 2:
            padding_h, padding_w = padding
        else:
            raise TypeError("Unacceptable padding type")
    elif padding == 'none':
        padding_h, padding_w = 0, 0
    elif padding == 'same':
        if filter_h % 2 == 0 or filter_w % 2 == 0:
            raise TypeError("Filter size could not keep same shape")
        padding_h, padding_w = filter_w // 2, filter_h // 2
    else:
        padding_h = padding_w = padding
    if not isinstance(padding_h, int) or not isinstance(padding_w, int):
        raise TypeError("Unacceptable padding type")
    return padding_h, padding_w


def img2col(image_array, filter_shape=(3, 3), stride=(1, 1), padding=(0, 0)):
    """
    :param image_array: shape = (input_number, input_channels, input_height, input_width)
    :param filter_shape: (filters_height, filters_width)
    :param stride: tuple means the different strides in each dimension
    :param padding: tuple means the different paddings in each dimension
    :return: shape = (input_number, output_height, output_width, input_channels * filters_height * filters_width)
    """
    n, c, h, w = image_array.shape
    filter_h, filter_w = filter_shape
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    out_h = (h + 2 * padding_h - filter_h) // stride_h + 1
    out_w = (w + 2 * padding_w - filter_w) // stride_w + 1

    if padding_h or padding_w:
        img = np.pad(image_array, ((0, 0), (0, 0), (padding_h, padding_h), (padding_w, padding_w)), 'constant')
    else:
        img = image_array
    col = np.empty((n, c, filter_h, filter_w, out_h, out_w))
    for y in range(filter_h):
        y_max = y + stride_h * out_h
        for x in range(filter_w):
            x_max = x + stride_w * out_w
            col[:, :, y, x, :, :] = img[:, :, y: y_max: stride_h, x: x_max: stride_w]

    return col.transpose(0, 4, 5, 1, 2, 3).reshape((n, out_h, out_w, -1))


def col2img(column_array, image_size=(1, 3, 32, 32), filter_shape=(3, 3), stride=(1, 1), padding=(0, 0)):
    """
    :param column_array: shape = (input_number * output_height * output_width, input_channels * filters_height * filters_width)
    :param image_size: (input_number, input_channels, input_height, input_width)
    :param filter_shape: (output_channels, input_channels, filters_height, filters_width)
    :param stride: tuple means the different strides in each dimension
    :param padding: tuple means the different paddings in each dimension
    :return: shape = (number of images, channels of images, height of images, width of images)
    """
    n, c, h, w = image_size
    filter_h, filter_w = filter_shape
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    out_h = (h + 2 * padding_h - filter_h) // stride_h + 1
    out_w = (w + 2 * padding_w - filter_w) // stride_w + 1
    column_array = column_array.reshape((n, out_h, out_w, c, filter_h, filter_w)).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((n, c, h + 2 * padding_h + stride_h - 1, w + 2 * padding_w + stride_w - 1))
    for y in range(filter_h):
        y_max = y + stride_h * out_h
        for x in range(filter_w):
            x_max = x + stride_w * out_w
            img[:, :, y: y_max: stride_h, x: x_max: stride_w] += column_array[:, :, y, x, :, :]
    return img[:, :, padding_h: padding_h + h, padding_w: padding_w + w]


def im2col(x: Tensor, filter_shape=(3, 3), stride=1, padding=0):
    """
    :param x: shape = (N, IC, IH, IW)
    :param filter_shape: (FH, FW)
    :param stride: integer means the same strides in all dimension,
                   tuple means the different strides in each dimension
    :param padding: integer means the same paddings in all dimension,
                    tuple means the different paddings in each dimension,
                    string 'same' means keep the image shape,
                    string 'none' means 0 in all dimension
    :return: shape = (N, OH, OW, IC * FH * FW)
    """
    n, c, h, w = x.shape
    stride_h, stride_w = get_stride_dim(stride)
    filter_h, filter_w = filter_shape
    padding_h, padding_w = get_padding_dim(padding, filter_h, filter_w)
    stride = (stride_h, stride_w)
    padding = (padding_h, padding_w)

    if (h + 2 * padding_h - filter_h) % stride_h or (w + 2 * padding_w - filter_w) % stride_w:
        raise ValueError("Unsuitable filter shape")

    y = img2col(x.data, filter_shape, stride, padding)
    if not x.requires_grad:
        return Tensor(y)

    def grad_fn(grad):
        return col2img(grad, x.shape, filter_shape, stride, padding)

    return Tensor(
        data=y,
        requires_grad=True,
        depend_on=[(x, grad_fn)]
    )
