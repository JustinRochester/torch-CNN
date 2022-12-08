import os.path

from ..CNN.GPU_np import np
from .load_data import read_data
from ..CNN.NeuralNetwork import *


def learning_rate_function(lr, i, tot):
    return lr


class LeNet:
    def __init__(self, learning_rate=1e-3, class_num=10):
        self.c, self.h, self.w = 1, 28, 28
        self.nn = NeuralNetwork((self.c, self.h, self.w),
                                class_num=class_num,
                                learning_rate=learning_rate,
                                loss_name='cross_entropy_softmax',
                                optimizer_name='Adam',
                                alpha=0.00,
                                learning_rate_function=learning_rate_function)
        self.nn.add(
            BatchNormalization((1, 28, 28)),

            Sequential(
                Conv2D(input_size=(1, 28, 28), filter_size=(5, 5), filter_num=6, padding=2),
                BatchNormalization((6, 28, 28)),
                Activation(input_size=(6, 28, 28), activation_name='relu'),
                MaxPool2D(input_size=(6, 28, 28), pooling_size=(2, 2))
            ),

            Sequential(
                Conv2D(input_size=(6, 14, 14), filter_size=(5, 5), filter_num=16),
                BatchNormalization((16, 10, 10)),
                Activation(input_size=(16, 10, 10), activation_name='relu'),
                MaxPool2D(input_size=(16, 10, 10), pooling_size=(2, 2))
            ),

            Flatten((16, 5, 5)),
            Dropout((5 * 5 * 16, 1)),

            Sequential(
                Linear(input_size=5 * 5 * 16, output_size=120),
                BatchNormalization((120, 1)),
                Activation(input_size=(120, 1), activation_name='relu'),
                Dropout((120, 1))
            ),

            Sequential(
                Linear(input_size=120, output_size=84),
                BatchNormalization((84, 1)),
                Activation(input_size=(84, 1), activation_name='relu'),
                Dropout((84, 1))
            ),

            Linear(input_size=84, output_size=class_num)
        )
        self.nn.build_model(self.nn.optimizer)

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

