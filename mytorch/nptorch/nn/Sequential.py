from .base import *
from .Module import Module


class Sequential(Module):
    """
    Collect a set of modules and their repeat times.
    It collects by a tuple like (modules: Module, repeat times: int).
    Each int in the tuple, defaults by 1.
    """
    def __init__(self, *args):
        super().__init__()
        self.layer_count = 0
        self.add(*args)

    def add(self, *args):
        for element in args:
            if not isinstance(element, tuple):
                element = (element,)
            if len(element) < 2:
                element = (*element, 1)
            elif len(element) > 2:
                raise ValueError("Not an tuple of module and repeat times")
            if not isinstance(element[0], Module) or not isinstance(element[1], int):
                raise ValueError("Not an tuple of module and repeat times")

            for times in range(element[1]):
                self.layer_count += 1
                layer_name = 'layer{}'.format(self.layer_count)
                self.__setattr__(layer_name, element[0])

    def forward(self, x: Tensor):
        for layer in self.layer_list:
            x = layer(x)
        return x
