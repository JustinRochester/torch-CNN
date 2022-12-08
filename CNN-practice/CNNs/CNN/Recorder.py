import os
import pickle
import re
from .GPU_np import np


class Recorder:
    def __init__(self, save_num=3):
        self.save_path = None
        self.save_num = save_num

    def set_path(self, path=None):
        self.save_path = path

    def save(self, filename, data):
        if self.save_path is None:
            return
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        filename = os.path.join(self.save_path, filename + '.pkl')
        with open(filename, 'wb') as file:
            pickle.dump(data, file)

    def load(self, filename):
        filename = os.path.join(self.save_path, filename + '.pkl')
        data = None
        if self.save_path is None or not os.path.exists(filename):
            return data
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        return data

    def clear_version(self, version, filename):
        lst = os.listdir(self.save_path)
        del_lst = []
        for file in lst:
            if re.match(".*" + filename + ".*", file):
                del_lst.append(file)
        lst = del_lst
        for i in range(version-self.save_num+1, version+1):
            keep_filename = filename + '%d.pkl' % i
            if keep_filename in lst:
                lst.remove(keep_filename)
        for file in lst:
            os.remove(os.path.join(self.save_path, file))

    def save_version(self, version, data):
        self.clear_version(version=version, filename='weight')
        self.save(
            filename='weight%d' % version,
            data=data
        )

    def save_log(self, version, *data_list):
        self.clear_version(version=version, filename='log')
        self.save(
            filename='log%d' % version,
            data=data_list
        )

    def exists_best(self):
        filename = os.path.join(self.save_path, 'best.pkl')
        return os.path.exists(filename)

    def remove_best(self):
        filename = os.path.join(self.save_path, 'best.pkl')
        if os.path.exists(filename):
            return os.remove(filename)

    def save_best(self, data):
        self.save(
            filename='best',
            data=data
        )

    def load_version(self, version):
        data = self.load(
            filename='weight%d' % version
        )
        for i in range(len(data)):
            data[i] = np.asarray(data[i].tolist())
        return data

    def load_log(self, version):
        return self.load(
            filename='log%d' % version
        )

    def load_best(self):
        data = self.load(
            filename='best'
        )
        for i in range(len(data)):
            data[i] = np.asarray(data[i].tolist())
        return data
