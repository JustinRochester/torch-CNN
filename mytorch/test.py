from nptorch.GPU_np import np
import nptorch
import nptorch.functional as F
import nptorch.layer as L
from nptorch.loss.CrossEntropy import CrossEntropy
from nptorch.optim.Adam import Adam
from nptorch.Module import Module


class LeNet(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = L.Conv2d(3, 6, (5, 5))
        self.pooling1 = L.MaxPool2d((2, 2))
        self.conv2 = L.Conv2d(6, 16, (5, 5))
        self.pooling2 = L.MaxPool2d((2, 2))
        self.fc1 = L.Linear(16 * 5 * 5, 120)
        self.fc2 = L.Linear(120, 84)
        self.fc3 = L.Linear(84, 10)

    def forward(self, x: nptorch.Tensor):
        n = x.shape[0]
        x = F.relu(self.conv1(x))
        x = self.pooling1(x)
        x = F.relu(self.conv2(x))
        x = self.pooling2(x)
        x = x.reshape((n, -1))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    nn = LeNet()
    data_list = nn.get_data_list()
    nn.load_data_list(iter(data_list))
    data_line = []
    for e in data_list:
        assert isinstance(e, np.ndarray)
        data_line += e.reshape(-1).tolist()
    data_line.sort()
    print(data_line)
    print(len(data_line))
    exit(0)
