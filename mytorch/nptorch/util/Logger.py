import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter as clock


flame_length = 0.1


class Logger:
    def __init__(self, train_data_to_test=False):
        self.__train_data_to_test = train_data_to_test
        self.__showing = False
        self.__last_show_time = -1
        self.loss_log = []
        self.train_acc_log = []
        self.test_acc_log = []

    def append_loss(self, value):
        self.loss_log.append(value)
        self.update_show()

    def append_train_acc(self, value):
        if self.__train_data_to_test:
            self.train_acc_log.append(value)
            self.update_show()

    def append_test_acc(self, value):
        self.test_acc_log.append(value)
        self.update_show()

    def get_data(self):
        return self.loss_log, self.train_acc_log, self.test_acc_log

    def load_data(self, data):
        self.loss_log, self.train_acc_log, self.test_acc_log = data
        if not self.__train_data_to_test:
            self.test_acc_log = []

    def start_show(self):
        self.__showing = True
        self.__last_show_time = -1
        plt.ion()
        self.update_show()

    def update_show(self):
        if not self.__showing:
            return
        if self.__last_show_time != -1 and clock() - self.__last_show_time < flame_length:
            return
        self.__last_show_time = clock()
        plt.clf()
        if self.__train_data_to_test:
            show_list = [self.loss_log, self.train_acc_log, self.test_acc_log]
            title_list = ['loss_value', 'train_accuracy', 'test_accuracy']
        else:
            show_list = [self.loss_log, self.test_acc_log]
            title_list = ['loss_value', 'test_accuracy']
        for idx in range(1, len(show_list)+1):
            subplot = plt.subplot(len(show_list), 1, idx)
            subplot.set_title(title_list[idx-1])
            y_value = show_list[idx-1]
            x_value = np.arange(len(y_value))
            if idx > 1:
                subplot.scatter(x_value, y_value)
            subplot.plot(x_value, y_value, '-')
        plt.pause(flame_length)
        plt.ioff()

    def stop_show(self):
        self.__showing = False
        self.__last_show_time = -1
        plt.ioff()
        plt.show()
