from ..GPU_np import np
from .LossFunction import LossFunction
from ..Base.softmax import softmax

eps = 1e-50


class CrossEntropySoftmax(LossFunction):
    @staticmethod
    def loss(label, output, regular_loss):
        n = output.shape[0]
        label = label.reshape((n, -1))
        softmax_output = softmax(output).reshape((n, -1))
        loss_value = softmax_output[np.arange(n), np.argmax(label, axis=1)] + eps
        loss_value = -np.log(loss_value) + regular_loss
        grad_output = softmax_output - label
        return loss_value, grad_output.reshape((n, -1, 1))
