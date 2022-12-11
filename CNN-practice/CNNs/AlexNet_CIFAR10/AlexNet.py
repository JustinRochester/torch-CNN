import os.path

from ..CNN.GPU_np import np
from .load_data import read_data
from ..CNN.NeuralNetwork import *


def learning_rate_function(lr, i, tot):
    return lr


class AlexNet:
    def __init__(self, learning_rate=1e-3, class_num=10):
        self.c, self.h, self.w = 3, 32, 32
        self.nn = NeuralNetwork((self.c, self.h, self.w),
                                class_num=class_num,
                                learning_rate=learning_rate,
                                loss_name='cross_entropy_softmax',
                                optimizer_name='Adam',
                                alpha=0.00,
                                learning_rate_function=learning_rate_function)
        self.nn.add(
            BatchNormalization((3, 32, 32)),

            Sequential(
                Conv2D(input_size=(3, 32, 32), filter_size=(3, 3), filter_num=96, padding=1),
                BatchNormalization((96, 32, 32)),
                Activation(input_size=(96, 32, 32), activation_name='relu'),
                MaxPool2D(input_size=(96, 32, 32), pooling_size=(2, 2))
            ),

            Sequential(
                Conv2D(input_size=(96, 16, 16), filter_size=(3, 3), filter_num=256, padding=1),
                BatchNormalization((256, 16, 16)),
                Activation(input_size=(256, 16, 16), activation_name='relu'),
                MaxPool2D(input_size=(256, 16, 16), pooling_size=(2, 2))
            ),

            Sequential(
                Conv2D(input_size=(256, 8, 8), filter_size=(3, 3), filter_num=384, padding=1),
                BatchNormalization((384, 8, 8)),
                Activation(input_size=(384, 8, 8), activation_name='relu'),
                MaxPool2D(input_size=(384, 8, 8), pooling_size=(2, 2))
            ),

            Sequential(
                Conv2D(input_size=(384, 4, 4), filter_size=(3, 3), filter_num=384, padding=1),
                BatchNormalization((384, 4, 4)),
                Activation(input_size=(384, 4, 4), activation_name='relu'),
                MaxPool2D(input_size=(384, 4, 4), pooling_size=(2, 2))
            ),

            Sequential(
                Conv2D(input_size=(384, 2, 2), filter_size=(3, 3), filter_num=256, padding=1),
                BatchNormalization((256, 2, 2)),
                Activation(input_size=(256, 2, 2), activation_name='relu'),
                MaxPool2D(input_size=(256, 2, 2), pooling_size=(2, 2))
            ),

            Flatten((256, 1, 1)),
            Dropout((1 * 1 * 256, 1)),

            Sequential(
                Linear(input_size=1 * 1 * 256, output_size=4096),
                BatchNormalization((4096, 1)),
                Activation(input_size=(4096, 1), activation_name='relu'),
                Dropout((4096, 1))
            ),

            Sequential(
                Linear(input_size=4096, output_size=4096),
                BatchNormalization((4096, 1)),
                Activation(input_size=(4096, 1), activation_name='relu'),
                Dropout((4096, 1))
            ),

            Linear(input_size=4096, output_size=class_num)
        )
        self.nn.build_model(self.nn.optimizer)

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

