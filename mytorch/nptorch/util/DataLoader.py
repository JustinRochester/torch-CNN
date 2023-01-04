from ..GPU_np import np
from ..nn.base import *


class DataLoader:
    def __init__(self, *dataset, batch_size=16, shuffle=True):
        self.len = dataset[0].shape[0]
        for data in dataset:
            assert data.shape[0] == self.len
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.select_index = None
        self.select_position = 0

    def __iter__(self):
        if self.select_index is None:
            if self.shuffle:
                self.select_index = np.random.permutation(self.len)
            else:
                self.select_index = np.arange(self.len)
            self.select_position = 0
        return self

    def __next__(self):
        if self.select_position < self.len:
            l = self.select_position
            r = l + self.batch_size
            if r > self.len:
                r = self.len
            self.select_position = r
            return tuple(
                Tensor(data[self.select_index[l:r]]) for data in self.dataset
            )
        else:
            self.select_index = None
            raise StopIteration
