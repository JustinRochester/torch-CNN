from mytorch.nptorch.GPU_np import np
from ..base.Tensor import Tensor


def abs(x: Tensor):
    return Tensor(np.sign(x.data)) * x
