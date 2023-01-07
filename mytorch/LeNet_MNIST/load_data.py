import torchvision.datasets.mnist as mnist
from ..nptorch.GPU_np import np
import os


def class2onehot(x):
    n = x.shape[0]
    class_number = int(np.max(x) - np.min(x) + 1)
    y = np.zeros((n, class_number, 1))
    y[np.arange(n), x] = 1
    return y


def read_data():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/MNIST/")

    kind = 'train'
    images_path = os.path.join(path, "%s-images-idx3-ubyte" % kind)
    labels_path = os.path.join(path, "%s-labels-idx1-ubyte" % kind)
    train_images = np.array(mnist.read_image_file(images_path)).reshape((-1, 1, 28, 28))
    train_labels = np.array(mnist.read_label_file(labels_path), dtype=np.int)

    kind = 't10k'
    images_path = os.path.join(path, "%s-images-idx3-ubyte" % kind)
    labels_path = os.path.join(path, "%s-labels-idx1-ubyte" % kind)
    test_images = np.array(mnist.read_image_file(images_path)).reshape((-1, 1, 28, 28))
    test_labels = np.array(mnist.read_label_file(labels_path), dtype=np.int)

    return train_images, train_labels, test_images, test_labels
