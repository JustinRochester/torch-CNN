from .Tensor import Tensor


class Parameter(Tensor):
    """
    A subclass of Tensor.
    It will register in parameter_list if it's used in class Module.
    """

    def __init__(self, data, requires_grad=False, depend_on=[]):
        super().__init__(data, requires_grad, depend_on)
