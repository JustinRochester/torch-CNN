import numpy as np


def pooling(image_array, shape=(2, 2), func=np.max):
    '''
    :param image_array: shape = (number of images, channels of images, height of images, width of images)
    :param shape: (pooling height, pooling width)
    :param func: pooling function
    :return: new_image_array: shape = (number of images, channels of images, new height of images, new width of images)
    '''
    n, c, h, w = image_array.shape
    h_size = h // shape[0]
    w_size = w // shape[1]
    pooling_array = np.empty((n, c, h_size, w_size))
    for i in range(h_size):
        for j in range(w_size):
            pooling_array[:, :, i, j] = func(image_array[:, :, i: i + shape[0], j: j + shape[1]], axis=(2, 3))
    return pooling_array


def max_pooling(image_array, shape=(2, 2)):
    return pooling(image_array, shape, np.max)


def avg_pooling(image_array, shape=(2, 2)):
    return pooling(image_array, shape, np.average)