from mytorch.nptorch.GPU_np import np
from ..base import *
from .Layer import Layer


class Dropout(Layer):
    """
    randomly dropout the input
    """
    def __init__(self,
                 probability: Tensor = Tensor(0.5)):
        super().__init__()
        self.probability = probability

    def forward(self, x: Tensor):
        if self.mode == 'train':
            dropout_mask = Tensor(np.random.uniform(0, 1, x.shape) < self.probability.data)
            return x * dropout_mask
        else:
            return x
