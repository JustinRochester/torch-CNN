from ..CNN.GPU_np import np
import os


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def read_data():
    batch_size = 10000
    train_data = np.empty((batch_size * 5, 3072))
    train_labels = np.zeros((batch_size * 5, 10))
    CIFAR_DIR = 'E:\\subject\\torch-CNN\\CNN-practice\\data\\CIFAR10\\cifar-10-batches-py'
    for i in range(5):
        filename = 'data_batch_%d' % (i + 1)
        filename = os.path.join(CIFAR_DIR, filename)
        dict = unpickle(filename)
        train_data[i * batch_size: (i + 1) * batch_size, :] = np.asarray(dict[b'data'])
        train_labels[np.arange(i * batch_size, (i + 1) * batch_size), dict[b'labels']] = 1
    filename = 'test_batch'
    filename = os.path.join(CIFAR_DIR, filename)
    dict = unpickle(filename)
    test_data = np.asarray(dict[b'data'])
    test_labels = np.zeros((batch_size, 10))
    test_labels[np.arange(batch_size), dict[b'labels']] = 1
    return train_data.reshape((-1, 3, 32, 32)), train_labels.reshape((-1, 10, 1)),\
           test_data.reshape((-1, 3, 32, 32)), test_labels.reshape((-1, 10, 1))


if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = read_data()
    print(train_data.shape)
    print(train_labels.shape)
    print(test_data.shape)
    print(test_labels.shape)
