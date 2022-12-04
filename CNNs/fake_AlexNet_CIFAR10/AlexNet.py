import os.path

from ..CNN.GPU_np import np
from .load_data import read_data
from ..CNN.NeuralNetwork import NeuralNetwork


def learning_rate_function(lr, i, tot):
    return lr


class AlexNet:
    def __init__(self, learning_rate=1e-3):
        self.c, self.h, self.w = 3, 32, 32
        self.nn = NeuralNetwork((self.c, self.h, self.w),
                                learning_rate=learning_rate,
                                loss_name='cross_entropy_softmax',
                                optimizer_name='Adam',
                                alpha=0.00,
                                learning_rate_function=learning_rate_function)
        self.nn.add_BN()

        self.nn.add_conv((3, 3), 64, padding=1)
        self.nn.add_BN()
        self.nn.add_activation('relu')
        self.nn.add_max_pool((2, 2))

        self.nn.add_conv((3, 3), 192, padding=1)
        self.nn.add_BN()
        self.nn.add_activation('relu')
        self.nn.add_max_pool((2, 2))

        self.nn.add_conv((3, 3), 384, padding=1)
        self.nn.add_BN()
        self.nn.add_activation('relu')

        self.nn.add_conv((3, 3), 256, padding=1)
        self.nn.add_BN()
        self.nn.add_activation('relu')

        self.nn.add_conv((3, 3), 256, padding=1)
        self.nn.add_BN()
        self.nn.add_activation('relu')
        self.nn.add_max_pool((2, 2))

        self.nn.add_flatten()
        self.nn.add_dropout()

        self.nn.add_fc((4096, 1))
        self.nn.add_BN()
        self.nn.add_activation('relu')
        self.nn.add_dropout()

        self.nn.add_fc((4096, 1))
        self.nn.add_BN()
        self.nn.add_activation('relu')
        self.nn.add_dropout()

        self.nn.add_fc((10, 1))

    def test_accuracy(self, version=10, batch_size=1024):
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weight")
        train_images, train_labels, test_images, test_labels = read_data()
        test_images = test_images / 255.0
        self.nn.load(version, save_path)
        return self.nn.test_accuracy(test_images, test_labels, batch_size)

    def work(self, epoch=10, batch_size=1024, version=0):
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weight")
        train_images, train_labels, test_images, test_labels = read_data()
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        # train_images, train_labels = train_images[:1000], train_labels[:1000]
        # test_images, test_labels = test_images[:1000], test_labels[:1000]
        self.nn.train(
            image_array=train_images,
            label_array=train_labels,
            epoch_number=epoch,
            batch_size=batch_size,
            test_image_array=test_images,
            test_label_array=test_labels,
            version=version,
            save_path=save_path)

