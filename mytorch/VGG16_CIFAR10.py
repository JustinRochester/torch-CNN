import os
from time import perf_counter as clock
import nptorch
from nptorch.GPU_np import np
from nptorch import nn
from nptorch.util import DataLoader, Recorder
from nptorch.nn import functional as F
from load_CIFAR10 import read_data
from train import train_epoch, evaluate, get_data, train


bias_need = False


class VGGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(3)

        self.conv_seq1 = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), padding=1, bias=bias_need),
            # nn.BatchNorm2d(64),
            nn.ReLu(),
            nn.Conv2d(64, 64, (3, 3), padding=1, bias=bias_need),
            # nn.BatchNorm2d(64),
            nn.ReLu(),
            nn.MaxPool2d((2, 2)),
        )

        self.conv_seq2 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), padding=1, bias=bias_need),
            # nn.BatchNorm2d(128),
            nn.ReLu(),
            nn.Conv2d(128, 128, (3, 3), padding=1, bias=bias_need),
            # nn.BatchNorm2d(128),
            nn.ReLu(),
            nn.MaxPool2d((2, 2)),
        )

        self.conv_seq3 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), padding=1, bias=bias_need),
            # nn.BatchNorm2d(256),
            nn.ReLu(),
            nn.Conv2d(256, 256, (3, 3), padding=1, bias=bias_need),
            # nn.BatchNorm2d(256),
            nn.ReLu(),
            nn.Conv2d(256, 256, (3, 3), padding=1, bias=bias_need),
            # nn.BatchNorm2d(256),
            nn.ReLu(),
            nn.MaxPool2d((2, 2)),
        )

        self.conv_seq4 = nn.Sequential(
            nn.Conv2d(256, 512, (3, 3), padding=1, bias=bias_need),
            # nn.BatchNorm2d(512),
            nn.ReLu(),
            nn.Conv2d(512, 512, (3, 3), padding=1, bias=bias_need),
            # nn.BatchNorm2d(512),
            nn.ReLu(),
            nn.Conv2d(512, 512, (3, 3), padding=1, bias=bias_need),
            # nn.BatchNorm2d(512),
            nn.ReLu(),
            nn.MaxPool2d((2, 2)),
        )

        self.conv_seq5 = nn.Sequential(
            nn.Conv2d(512, 512, (3, 3), padding=1, bias=bias_need),
            # nn.BatchNorm2d(512),
            nn.ReLu(),
            nn.Conv2d(512, 512, (3, 3), padding=1, bias=bias_need),
            # nn.BatchNorm2d(512),
            nn.ReLu(),
            nn.Conv2d(512, 512, (3, 3), padding=1, bias=bias_need),
            # nn.BatchNorm2d(512),
            nn.ReLu(),
            nn.MaxPool2d((2, 2)),
        )

        self.dropout = nn.Dropout()
        self.fc_seq1 = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096, bias=bias_need),
            # nn.BatchNorm1d(512),
            nn.ReLu(),
            nn.Dropout(),
        )

        self.fc_seq2 = nn.Sequential(
            nn.Linear(4096, 4096, bias=bias_need),
            # nn.BatchNorm1d(512),
            nn.ReLu(),
            nn.Dropout(),
        )

        self.fc_output = nn.Linear(4096, 10)

    def forward(self, x: nptorch.Tensor):
        n = x.shape[0]
        x = self.bn0(x)
        x = self.conv_seq1(x)
        x = self.conv_seq2(x)
        x = self.conv_seq3(x)
        x = self.conv_seq4(x)
        x = self.conv_seq5(x)
        x = x.reshape((n, -1))
        x = self.dropout(x)
        x = self.fc_seq1(x)
        x = self.fc_seq2(x)
        x = self.fc_output(x)
        return x


def work(version=0, epoch_number=100):
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
    net = VGGNet()

    # set dataset
    train_evaluate_data, test_evaluate_data, train_data = get_data(
        read_data=read_data,
        class_num=10,
        train_batch=64,
        test_batch=128,
        train_data_to_test=False,
    )

    # set loss function
    loss_function = nn.LossCollector(
        nn.CrossEntropySoftmax_Loss(),
        (nn.Regular_2_Loss(), 1e-4)
    )

    # set optimizer
    optimizer = nn.Adam(
        parameter_list=net.parameters(),
        learning_rate=1e-4,
        learning_rate_function=(lambda x: x*0.99)
    )

    # training
    train(
        net=net,
        loss_function=loss_function,
        optimizer=optimizer,
        train_data=train_data,
        recorder=recorder,
        test_data=(train_evaluate_data, test_evaluate_data),
        train_data_to_test=False,
        version=version,
        epoch_number=epoch_number,
    )


if __name__ == '__main__':
    work()
    exit(0)
