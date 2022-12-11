from CNNs.CNN.GPU_np import np
from CNNs.AlexNet_CIFAR10.AlexNet import AlexNet


if __name__ == '__main__':
    nn = AlexNet(
        learning_rate=1e-1
    )
    nn.work(
        epoch=10,
        batch_size=16,
        version=0
    )
