import numpy as np
from collections import Counter


def L1_dis(samples, sample):
    return np.sum(np.abs(samples - sample), axis=1)


def L2_dis(samples, sample):
    return np.sum(np.power(samples - sample, 2), axis=1)**0.5


class KNN:
    def __init__(self, k=1, dis_func=L1_dis):
        self.xtr = None
        self.ytr = None
        self.k = k
        self.dis_func = dis_func

    def train(self, x, y):
        self.xtr = x
        self.ytr = y

    def predict(self, x):
        num_test = x.shape[0]
        y_pred = np.zeros(num_test, dtype=self.ytr.dtype)

        for i in range(num_test):
            distances = self.dis_func(self.xtr, x[i])
            distances = np.argsort(distances)[:self.k]
            labels = self.ytr[distances]

            index = Counter(labels).most_common()[0]
            y_pred[i] = index[0]

        return y_pred


if __name__ == '__main__':
    n = 1000
    C = 10
    train_x = np.array([np.random.rand(2) for i in range(n)])
    train_y = np.array([np.random.randint(1, C + 1) for i in range(n)])

    nn1 = KNN(100, dis_func=L1_dis)
    nn1.train(train_x, train_y)

    nn2 = KNN(100, dis_func=L2_dis)
    nn2.train(train_x, train_y)

    test = np.array([np.random.rand(2) for i in range(10)])
    y1 = nn1.predict(test)
    y2 = nn2.predict(test)
    print(y1)
    print(y2)
