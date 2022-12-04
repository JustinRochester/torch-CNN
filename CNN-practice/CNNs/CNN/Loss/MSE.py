from ..GPU_np import np
from .LossFunction import LossFunction


class MSE(LossFunction):
    @staticmethod
    def loss(label, output, regular_loss):
        n = output.shape[0]
        loss_value = np.sqrt(np.sum(np.square(output - label), axis=(1, 2))) + regular_loss
        grad_output = (output - label) / loss_value.reshape((n, 1, 1))
        return loss_value, grad_output
