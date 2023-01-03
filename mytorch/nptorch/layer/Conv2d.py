from ..GPU_np import np
from ..base import *
from ..functional.im2col import im2col
from .Layer import Layer


class Conv2d(Layer):
    """
    Input-tensor shape: [N, IC, IH, IW] => x;
    Output-tensor shape: [N, OC, OH, OW] => y;
    Transform x by Linear Transforms: y = x * F + b;
    Parameter-F shape: [OC, IC, FH, FW];
    Parameter-b shape: [1, OC];
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple,
                 stride=1,
                 padding=0,
                 bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        filters_initializer = Initializer(
            initial_mu=0,
            initial_std=np.sqrt(2 / in_channels),
        )
        self.filters = Parameter(
            data=filters_initializer(shape=(out_channels, in_channels) + kernel_size),
            requires_grad=True,
        )
        self.im2col = lambda x: im2col(x, filter_shape=kernel_size, stride=stride, padding=padding)
        self.bias = None

        if not bias:
            return
        bias_initializer = Initializer(
            initial_mu=0,
            initial_std=np.sqrt(2 / in_channels),
        )
        self.bias = Parameter(
            data=bias_initializer(shape=(1, out_channels)),
            requires_grad=True,
        )

    def __call__(self, x: Tensor, *args, **kwargs):
        x = self.im2col(x)
        n, oh, ow, _ = x.shape
        x = x.reshape((n * oh * ow, -1)) @ self.filters.reshape((self.out_channels, -1)).T()
        if self.bias is not None:
            x += self.bias
        return x.reshape((n, oh, ow, -1)).transpose(0, 3, 1, 2)
