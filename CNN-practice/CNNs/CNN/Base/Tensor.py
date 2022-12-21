from ..GPU_np import np
from ..Interfaces.Savable import Savable


class Tensor(Savable):
    """
    Structure for each numpy tensor in the neural network.
    Tensor.value shows the value of current Tensor.
    Tensor.grad shows the cumulative gradient of current Tensor
    """
    def __init__(self, shape=(1,), initial_mu=0, initial_std=1):
        """
        Initialize the Tensor's value with normal distribution N(initial_mu, initial_std).
        Initialize the Tensor with constant value C with parameters like(initial_mu = C, initial_std=0).
        """
        self.shape = shape
        self.value = np.random.normal(initial_mu, initial_std, shape)
        self.grad = np.empty(shape)

    def set_data(self, data_iter):
        """
        Set the Tensor's value and gradient with the data in data_iter.
        """
        self.value = next(data_iter)
        self.grad = next(data_iter)

    def get_data(self):
        """
        Get the Tensor's value and gradient.
        """
        return [self.value, self.grad]

    def zero_grad(self):
        """
        Clear the Tensor's current gradient.
        """
        self.grad = np.zeros_like(self.grad)

    def multi_grad(self, multiply=1):
        """
        Multiply the Tensor's gradient with parameter multiply.
        Average the batch's gradient with parameters like (multiply = 1 / batch_size)
        """
        self.grad *= multiply
