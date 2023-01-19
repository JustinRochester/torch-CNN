import os
from time import perf_counter as clock
import nptorch
from nptorch.GPU_np import np
from nptorch import nn
from nptorch.util import DataLoader, Recorder
from nptorch.nn import functional as F
from load_MNIST import read_data
from train import train_epoch, evaluate, get_data, train


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(1)

        self.conv_seq1 = nn.Sequential(
            nn.Conv2d(1, 6, (5, 5), padding=2, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLu(),
            nn.MaxPool2d((2, 2)),
        )

        self.conv_seq2 = nn.Sequential(
            nn.Conv2d(6, 16, (5, 5), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLu(),
            nn.MaxPool2d((2, 2)),
        )

        self.fc_seq1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120, bias=False),
            nn.BatchNorm1d(120),
            nn.Dropout(),
            nn.ReLu(),
        )

        self.fc_seq2 = nn.Sequential(
            nn.Linear(120, 84, bias=False),
            nn.BatchNorm1d(84),
            nn.Dropout(),
            nn.ReLu(),
        )

        self.fc_output = nn.Linear(84, 10)

    def forward(self, x: nptorch.Tensor):
        n = x.shape[0]
        x = self.bn0(x)
        x = self.conv_seq1(x)
        x = self.conv_seq2(x)
        x = x.reshape((n, -1))
        x = self.fc_seq1(x)
        x = self.fc_seq2(x)
        x = self.fc_output(x)
        return x


def work(version=0, epoch_number=10):
    # set recorder
    recorder = Recorder()
    recorder_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "weight",
        ".".join(
            os.path.split(__file__)[-1].split('.')[:-1]
        ),
    )
    recorder.set_path(recorder_path)

    # set neural network
    net = LeNet()

    # set dataset
    train_evaluate_data, test_evaluate_data, train_data = get_data(
        read_data=read_data,
        class_num=10,
        train_batch=4096,
        test_batch=8192,
        train_data_to_test=True,
    )

    # set loss function
    loss_function = nn.LossCollector(
        nn.CrossEntropySoftmax_Loss(),
        (nn.Regular_2_Loss(net.parameters()), 1e-3)
    )

    # set optimizer
    optimizer = nn.Adam(net.parameters(), learning_rate=1e-3)

    # training
    train(
        net=net,
        loss_function=loss_function,
        optimizer=optimizer,
        train_data=train_data,
        recorder=recorder,
        test_data=(train_evaluate_data, test_evaluate_data),
        train_data_to_test=True,
        version=0,
        epoch_number=10,
    )


if __name__ == '__main__':
    work()
    exit(0)
