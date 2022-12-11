import abc


class Propagable:
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
