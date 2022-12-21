import abc


class Propagatable:
    """
    This interface regulates the class which could propagate in this neural network.
    Class which implements this interface could use predict_forward to evaluate data, use forward and backward to train.
    Method zero_grad and multi_grad is used to clear the cumulative gradient and multiply it by a constant number.
    """
    @abc.abstractmethod
    def predict_forward(self, input_value):
        pass

    @abc.abstractmethod
    def forward(self, input_value):
        pass

    @abc.abstractmethod
    def backward(self, output_grad):
        pass

    @abc.abstractmethod
    def zero_grad(self):
        pass

    @abc.abstractmethod
    def multi_grad(self, multiply=1):
        pass
