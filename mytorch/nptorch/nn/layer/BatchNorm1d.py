from mytorch.nptorch.GPU_np import np
from ..base import *
from ..functional import batch_normalization, sqrt
from .Layer import Layer


eps = 1e-8


class BatchNorm1d(Layer):
    """
    Input-tensor shape: [N, C] => x;
    sampling-mu = mean(x, axis=0, keepdims=1)
    sampling-sigma2 = mean( square(x - sampling-mu), axis=0, keepdims=1)
    """
    def __init__(self,
                 features: int,
                 rho: Tensor = Tensor(0.9)):
        super().__init__()
        self.features = features
        self.rho = rho
        zeros_initializer = Initializer(
            initial_mu=0,
            initial_std=0,
        )
        self.running_mu = Parameter(zeros_initializer(shape=(1, features)))
        ones_initializer = Initializer(
            initial_mu=1,
            initial_std=0,
        )
        self.running_sigma2 = Parameter(ones_initializer(shape=(1, features)))
        self.gamma = Parameter(
            data=ones_initializer(shape=(1, features)),
            requires_grad=True,
        )
        self.beta = Parameter(
            data=zeros_initializer(shape=(1, features)),
            requires_grad=True,
        )

    def forward(self, x: Tensor):
        if self.mode == 'train':
            x, sampling_mu, sampling_sigma2 = batch_normalization(x, axis=0)
            self.running_mu = self.running_mu * (Tensor(1) - self.rho) + sampling_mu * self.rho
            self.running_sigma2 = self.running_sigma2 * (Tensor(1) - self.rho) + sampling_sigma2 * self.rho
        else:
            x = (x - self.running_mu) / sqrt(self.running_sigma2 + eps)
        return x * self.gamma + self.beta
