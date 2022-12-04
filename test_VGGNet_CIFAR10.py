from CNNs.CNN.GPU_np import np
from CNNs.VGG16_CIFAR10.VGGNet import VGGNet


if __name__ == '__main__':
    nn = VGGNet(
        learning_rate=1e-1
    )
    nn.work(
        epoch=84,
        batch_size=256,
        version=84
    )
