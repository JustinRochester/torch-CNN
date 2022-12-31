from nptorch.GPU_np import np
import nptorch
import nptorch.functional as F
import nptorch.layer as L

conv1 = L.Conv2d(3, 6, (5, 5))
pooling1 = L.MaxPool2d((2, 2))
conv2 = L.Conv2d(6, 16, (5, 5))
pooling2 = L.MaxPool2d((2, 2))
fc1 = L.Linear(16 * 5 * 5, 120)
fc2 = L.Linear(120, 84)
fc3 = L.Linear(84, 10)


def forward(x: nptorch.Tensor):
    n = x.shape[0]
    x = F.relu(conv1(x))
    x = pooling1(x)
    x = F.relu(conv2(x))
    x = pooling2(x)
    x = x.reshape((n, -1))
    x = F.relu(fc1(x))
    x = F.relu(fc2(x))
    x = fc3(x)
    return x


if __name__ == '__main__':
    x = nptorch.Tensor(np.random.random((512, 3, 32, 32)), requires_grad=True)
    y = forward(x)
    print(y.shape)
    y.backward()
    print(x.shape)
