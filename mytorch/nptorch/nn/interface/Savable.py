import abc


class Savable:
    @abc.abstractmethod
    def get_data_list(self):
        pass

    @abc.abstractmethod
    def load_data_list(self, data_iter):
        pass
