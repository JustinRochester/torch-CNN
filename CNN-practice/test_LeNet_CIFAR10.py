from CNNs.CNN.GPU_np import np
from CNNs.LeNet_CIFAR10.LeNet import LeNet
import sys


if __name__ == '__main__':
    nn = LeNet(
        learning_rate=1e-1
    )
    nn.work(
        epoch=10,
        batch_size=1024,
        run_size=32,
        version=7
    )
