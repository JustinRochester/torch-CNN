from CNNs.CNN.GPU_np import np
from CNNs.LeNet_CIFAR10.LeNet import LeNet
import sys


if __name__ == '__main__':
    try:
        nn = LeNet(
            learning_rate=1e-3
        )
        nn.work(
            epoch=600,
            batch_size=1024,
            version=600
        )
    finally:
        sys.exit(0)
