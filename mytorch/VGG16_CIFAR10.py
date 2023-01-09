import os
from time import perf_counter as clock
import nptorch
from nptorch.GPU_np import np
from nptorch import nn
from nptorch.util import DataLoader, Recorder
from nptorch.nn import functional as F
from load_CIFAR10 import read_data


class VGGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(3)

        self.conv_seq1 = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLu(),
            nn.Conv2d(64, 64, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLu(),
            nn.MaxPool2d((2, 2)),
        )

        self.conv_seq2 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLu(),
            nn.Conv2d(128, 128, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLu(),
            nn.MaxPool2d((2, 2)),
        )

        self.conv_seq3 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLu(),
            nn.Conv2d(256, 256, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLu(),
            nn.Conv2d(256, 256, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLu(),
            nn.MaxPool2d((2, 2)),
        )

        self.conv_seq4 = nn.Sequential(
            nn.Conv2d(256, 512, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLu(),
            nn.Conv2d(512, 512, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLu(),
            nn.Conv2d(512, 512, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLu(),
            nn.MaxPool2d((2, 2)),
        )

        self.conv_seq5 = nn.Sequential(
            nn.Conv2d(512, 512, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLu(),
            nn.Conv2d(512, 512, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLu(),
            nn.Conv2d(512, 512, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLu(),
            nn.MaxPool2d((2, 2)),
        )

        self.fc_seq1 = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096, bias=False),
            nn.Dropout(),
            nn.BatchNorm1d(4096),
            nn.ReLu(),
        )

        self.fc_seq2 = nn.Sequential(
            nn.Linear(4096, 4096, bias=False),
            nn.Dropout(),
            nn.BatchNorm1d(4096),
            nn.ReLu(),
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
        x = self.fc_seq1(x)
        x = self.fc_seq2(x)
        x = self.fc_output(x)
        return x


def train_epoch(
        turn: int, total_turn: int,
        net: nn.Module,
        loss_function: nn.LossFunction,
        optimizer: nn.Optimizer,
        train_data: DataLoader):
    epoch_start_time = clock()
    net.train_mode()
    finished = 0
    length = train_data.len
    loss_list = []
    for images, labels in train_data:
        now = train_data.select_position
        predict = net(images)
        loss = loss_function(predict, labels)
        loss.backward()
        optimizer.step()
        loss.zero_grad()
        average_loss = float(loss.data) / (now - finished)
        loss_list.append(average_loss)
        print('epoch {}/{} batch {}/{}, loss={}'.format(turn, total_turn, now, length, average_loss))
        finished = now
    print('epoch {}/{} cost time: {} second(s)'.format(turn, total_turn, clock() - epoch_start_time))
    return loss_list


def evaluate(net, data):
    net.predict_mode()
    acc = 0
    for images, labels in data:
        predict = net(images).data
        predict = predict.argmax(axis=1)
        acc += np.sum(predict == labels.data)
    return acc * 100 / data.len


def work():
    net = VGGNet()
    recorder = Recorder()
    recorder_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "weight",
        ".".join(
            os.path.split(__file__)[-1].split('.')[:-1]
        ),
    )
    recorder.set_path(recorder_path)

    train_images, train_labels, test_images, test_labels = read_data()
    test_data = DataLoader(test_images, test_labels, batch_size=256, shuffle=False)

    class_num = 10
    tmp = np.zeros((train_images.shape[0], class_num))
    tmp[np.arange(train_images.shape[0]), train_labels] = 1
    train_labels = tmp
    train_data = DataLoader(train_images, train_labels, batch_size=128)

    loss_function = nn.LossCollector(
        nn.CrossEntropySoftmax_Loss(),
    )
    optimizer = nn.Adam(net.parameters(), learning_rate=1e-3)

    epoch_number = 10

    for t in range(1, epoch_number + 1):
        train_epoch(t, epoch_number, net, loss_function, optimizer, train_data)
        recorder.save_version(
            version=t,
            data=net.get_data_list() + optimizer.get_data_list()
        )

        now = clock()
        test_acc = evaluate(net, test_data)
        print('cost time in predict test_data: {} second(s)'.format(clock() - now))

        print('epoch {}/{}, train data accuracy={}, test data accuracy={}'.format(t, epoch_number, train_acc, test_acc))


if __name__ == '__main__':
    work()
    exit(0)
