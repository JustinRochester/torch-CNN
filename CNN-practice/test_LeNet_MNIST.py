from CNNs.LeNet_MNIST.LeNet import LeNet


if __name__ == '__main__':
    nn = LeNet(
        learning_rate=1e-1
    )
    nn.work(
        epoch=10,
        batch_size=4096,
        run_size=1024,
        version=0
    )
