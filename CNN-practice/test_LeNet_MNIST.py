from CNNs.LeNet_MNIST.LeNet import LeNet


if __name__ == '__main__':
    nn = LeNet(
        learning_rate=1e-1
    )
    nn.work(
        epoch=15,
        batch_size=2048,
        version=10
    )
