from mytorch import nptorch
from nptorch.GPU_np import np
from nptorch import nn
from nptorch.util import DataLoader
from nptorch.nn import functional as F
from load_MNIST import read_data


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 6, (5, 5), padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(6)
        self.pooling1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(6, 16, (5, 5), bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pooling2 = nn.MaxPool2d((2, 2))
        self.fc1 = nn.Linear(16 * 5 * 5, 120, bias=False)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84, bias=False)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: nptorch.Tensor):
        n = x.shape[0]
        x = self.bn0(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pooling1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pooling2(x)
        x = x.reshape((n, -1))
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        return x


def work():
    net = LeNet()

    train_images, train_labels, test_images, test_labels = read_data()
    class_num = 10

    tmp = np.zeros((train_images.shape[0], class_num))
    tmp[np.arange(train_images.shape[0]), train_labels] = 1
    train_labels = tmp

    train_data = DataLoader(train_images, train_labels, batch_size=1024)
    test_data = DataLoader(test_images, test_labels, batch_size=1024, shuffle=False)

    loss_function = nn.CrossEntropy()
    optimizer = nn.Adam(net.parameters(), learning_rate=1e-3)

    epoch_number = 5

    for t in range(epoch_number):
        finished = 0
        for images, labels in train_data:
            now = train_data.select_position
            predict = net.forward(images)
            loss = loss_function(predict, labels)
            loss.backward()
            optimizer.step()
            loss.zero_grad()
            print('epoch {} batch {}, loss={}'.format(t + 1, now, float(loss.data) / (now - finished)))
            finished = now
        acc = 0
        for images, labels in test_data:
            predict = net.forward(images).data
            predict = predict.argmax(axis=1)
            acc += np.sum(predict == labels.data)
        print('epoch {}, acc={}%'.format(t + 1, acc * 100 / test_data.len))


if __name__ == '__main__':
    work()
    exit(0)
