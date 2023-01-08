import os
from time import perf_counter as clock
from mytorch import nptorch
from nptorch.GPU_np import np
from nptorch import nn
from nptorch.util import DataLoader, Recorder
from nptorch.nn import functional as F
from load_MNIST import read_data


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
            nn.Dropout(),
            nn.BatchNorm1d(120),
            nn.ReLu(),
        )

        self.fc_seq2 = nn.Sequential(
            nn.Linear(120, 84, bias=False),
            nn.Dropout(),
            nn.BatchNorm1d(84),
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
    net = LeNet()
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
    train_evaluate_data = DataLoader(train_images, train_labels, batch_size=2048, shuffle=False)
    test_data = DataLoader(test_images, test_labels, batch_size=2048, shuffle=False)

    class_num = 10
    tmp = np.zeros((train_images.shape[0], class_num))
    tmp[np.arange(train_images.shape[0]), train_labels] = 1
    train_labels = tmp
    train_data = DataLoader(train_images, train_labels, batch_size=1024)

    loss_function = nn.LossCollector(
        nn.CrossEntropySoftmax_Loss(),
        (nn.Regular_2_Loss(net.parameters()), 1e-1)
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
        train_acc = evaluate(net, train_evaluate_data)
        print('cost time in predict train_data: {} second(s)'.format(clock() - now))

        now = clock()
        test_acc = evaluate(net, test_data)
        print('cost time in predict test_data: {} second(s)'.format(clock() - now))

        print('epoch {}/{}, train data accuracy={}, test data accuracy={}'.format(t, epoch_number, train_acc, test_acc))


if __name__ == '__main__':
    work()
    exit(0)
