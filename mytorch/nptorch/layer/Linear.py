from ..GPU_np import np
from ..base import *
from .Layer import Layer


class Linear(Layer):
    """
    Input-tensor shape: [N, I] => x;
    Output-tensor shape: [N, O] => y;
    Transform x by Linear Transforms: y = x @ W + b;
    Parameter-W shape: [I, O];
    Parameter-b shape: [1, O];
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        weight_initializer = Initializer(
            initial_mu=0,
            initial_std=np.sqrt(2 / in_features),
        )
        self.weight = Parameter(
            data=weight_initializer(shape=(in_features, out_features)),
            requires_grad=True,
        )
        self.bias = None

        if not bias:
            return
        bias_initializer = Initializer(
            initial_mu=0,
            initial_std=np.sqrt(2 / in_features),
        )
        self.bias = Parameter(
            data=bias_initializer(shape=(1, out_features)),
            requires_grad=True,
        )

    def __call__(self, x: Tensor, *args, **kwargs):
        x @= self.weight
        if self.bias is not None:
            x += self.bias
        return x
