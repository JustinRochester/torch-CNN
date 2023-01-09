from ..GPU_np import np


class Initializer:
    """
    An initializer for Tensor.
    It will produce Tensor's data with inputting shape.
    Each unit producing by it obeys normal distribution with parameter N(mu, std**2).
    Especially, it will produce data full of constant value if std==0.
    It will produce data obeyed standard normal distribution by default.
    """
    def __init__(self, initial_mu=0, initial_std=1):
        self.mu = initial_mu
        self.std = initial_std

    def __call__(self, shape=(1,), *args, **kwargs):
        return np.random.normal(self.mu, self.std, shape)
