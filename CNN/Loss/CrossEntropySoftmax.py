from ..GPU_np import np
from .LossFunction import LossFunction
from ..Base.softmax import softmax


class CrossEntropySoftmax(LossFunction):
    @staticmethod
    def loss(label, output, regular_loss):
        n = output.shape[0]
        label = label.reshape((n, -1))
        softmax_output = softmax(output).reshape((n, -1))
        loss_value = - np.sum(label * np.log(softmax_output), axis=1) + regular_loss
        grad_output = np.sum(label, axis=1).reshape((n, 1)) * softmax_output - label
        return loss_value, grad_output.reshape((n, -1, 1))
