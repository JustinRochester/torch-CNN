from .base import *


class Module:
    def __init__(self):
        self.parameter_list = []
        self.save_list = []

    def parameters(self):
        return self.parameter_list

    def __setattr__(self, key, value):
        self.__dict__[key] = value
        if type(value) is Parameter:
            self.parameter_list.append(value)
            self.save_list.append(value)
        elif isinstance(value, Module):
            self.parameter_list += value.parameter_list
            self.save_list.append(value)

    def get_data_list(self):
        data_list = []
        for save_element in self.save_list:
            if isinstance(save_element, Module):
                data_list += save_element.get_data_list()
            else:
                data_list.append(save_element.data)
        return data_list

    def load_data_list(self, data_iter):
        for save_element in self.save_list:
            if isinstance(save_element, Module):
                save_element.load_data_list(data_iter)
            else:
                save_element.data = next(data_iter)
