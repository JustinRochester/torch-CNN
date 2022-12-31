from ..GPU_np import np


class Tensor:
    """
    An automic unit in neural network.
    The class holds the data, the gradient value of this variable.
    And it holds some attribution to calculate backward propagation.
    """

    def __init__(self, data, requires_grad=False, depend_on=[]):
        """
        :param data: A ndarray in numpy or just a number,
                    which shows the initial value of this tensor.
        :param requires_grad: A boolean variable default by false.
                    It will calculate the gradient in forward when it's true.
        :param depend_on: It shows how this tensor is made by,
                    and the grad_fn hold by it will be used in backward propagation.
        """
        if not isinstance(data, np.ndarray):
            if not isinstance(data, list):
                data = [data]
            data = np.array(data, dtype=np.float32)
        self.data = data
        self.shape = data.shape
        self.requires_grad = requires_grad
        self.grad = np.zeros(self.shape)
        self._depend_on = depend_on

    def __str__(self):
        return 'Tensor shape : {}, data : {}'.format(self.shape, self.data)

    def reshape(self, shape):
        """
        :param shape: The shape this tensor will be changed to.
        :return: A new tensor with inputting shape.
        """
        if not self.requires_grad:
            return Tensor(self.data.reshape(shape))

        def grad_fn(grad):
            return grad.reshape(self.shape)

        return Tensor(
            data=self.data.reshape(shape),
            requires_grad=True,
            depend_on=[(self, grad_fn)]
        )

    def transpose(self, *axes):
        """
        :param axes: A 0-based permutation with length n, determined the turns of transposed tensor
        :return: A new tensor with inputting turn
        """
        n = len(axes)
        if len(self.shape) != n:
            raise TypeError("Unexpected axis")
        if not self.requires_grad:
            return Tensor(self.data.transpose(*axes))
        i_axis = [-1 for i in range(n)]
        for i in range(n):
            i_axis[axes[i]] = i
        i_axis = tuple(i_axis)

        def grad_fn(grad):
            return grad.transpose(i_axis)

        return Tensor(
            data=self.data.transpose(*axes),
            requires_grad=True,
            depend_on=[(self, grad_fn)]
        )

    def T(self):
        """
        :return: A new tensor with transposition.
        """
        if not self.requires_grad:
            return Tensor(self.data.T)

        def grad_fn(grad):
            return grad.T

        return Tensor(
            data=self.data.T,
            requires_grad=True,
            depend_on=[(self, grad_fn)]
        )

    def broadcast(self, shape):
        """
        :param shape: The shape which this tensor will broadcast to.
        :return: A new tensor with inputting shape.
        """
        if self.shape == shape:
            return self

        # keep dimensions of res is same to shape
        if len(self.shape) < len(shape):
            res = self.data.reshape((1,) * (len(shape) - len(self.shape)) + self.shape)
        elif len(self.shape) == len(shape):
            res = self.data
        else:
            raise TypeError("Unexpected broadcast-shape")

        # calculate which dimensions will be repeated
        tile_reps = []
        sum_axis = []
        for i in range(len(shape)):
            if shape[i] == res.shape[i]:
                tile_reps.append(1)
            elif res.shape[i] == 1:
                tile_reps.append(shape[i])
                sum_axis.append(i)
            else:
                raise TypeError("Unexpected broadcast-shape")
        tile_reps = tuple(tile_reps)
        sum_axis = tuple(sum_axis)

        if not self.requires_grad:
            return Tensor(np.tile(res, tile_reps))

        def grad_fn(grad):
            return np.sum(grad, axis=sum_axis).reshape(self.shape)

        return Tensor(
            data=np.tile(res, tile_reps),
            requires_grad=True,
            depend_on=[(self, grad_fn)]
        )

    @staticmethod
    def _get_broadcast_shape(self, other):
        """
        Get two tensors who will be calculated later after broadcast.
        Return the shape which this two tensors will broadcast to.
        """
        shape1, shape2 = self.shape, other.shape
        dif_len = len(shape1) - len(shape2)
        if dif_len < 0:
            shape1 = (1,) * (-dif_len) + shape1
        elif dif_len > 0:
            shape2 = (1,) * dif_len + shape2

        ret_shape = []
        for i in range(len(shape1)):
            ret_shape.append(max(shape1[i], shape2[i]))
        return tuple(ret_shape)

    def __add__(self, other):
        """
        y=a+b;
        dL/da=dL/db=dL/dy;
        """
        if not isinstance(other, Tensor):
            other = Tensor(other)
        shape = Tensor._get_broadcast_shape(self, other)
        lhs = self.broadcast(shape)
        rhs = other.broadcast(shape)

        def grad_fn(grad):
            return grad

        lst = []
        requires_grad = False
        if lhs.requires_grad:
            lst.append((lhs, grad_fn))
            requires_grad = True
        if rhs.requires_grad:
            lst.append((rhs, grad_fn))
            requires_grad = True
        return Tensor(
            data=lhs.data + rhs.data,
            requires_grad=requires_grad,
            depend_on=lst,
        )

    def __radd__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other + self

    def __neg__(self):
        """
        y=-a;
        dL/da=-dL/dy;
        """
        if not self.requires_grad:
            return Tensor(-self.data)

        def grad_fn(grad):
            return -grad

        return Tensor(
            data=-self.data,
            requires_grad=True,
            depend_on=[(self, grad_fn)]
        )

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other - self

    def __mul__(self, other):
        """
        y=a*b;
        dL/da=dL/dy*dy/da=dL/dy*b;
        dL/db=dL/dy*dy/db=dL/dy*a;
        """
        if not isinstance(other, Tensor):
            other = Tensor(other)
        shape = Tensor._get_broadcast_shape(self, other)
        lhs = self.broadcast(shape)
        rhs = other.broadcast(shape)

        def grad_fn_l(grad):
            return grad * rhs.data

        def grad_fn_r(grad):
            return grad * lhs.data

        lst = []
        requires_grad = False
        if lhs.requires_grad:
            lst.append((lhs, grad_fn_l))
            requires_grad = True
        if rhs.requires_grad:
            lst.append((rhs, grad_fn_r))
            requires_grad = True
        return Tensor(
            data=lhs.data * rhs.data,
            requires_grad=requires_grad,
            depend_on=lst,
        )

    def __rmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other * self

    def __truediv__(self, other):
        """
        y=a/b;
        dL/da=dL/dy*dy/da=dL/dy/b;
        dL/db=dL/dy*dy/db=dL/dy*a*(-1/b**2);
        """
        if not isinstance(other, Tensor):
            other = Tensor(other)
        shape = Tensor._get_broadcast_shape(self, other)
        lhs = self.broadcast(shape)
        rhs = other.broadcast(shape)

        def grad_fn_l(grad):
            return grad / rhs.data

        def grad_fn_r(grad):
            return - grad * lhs.data / (rhs.data * rhs.data)

        lst = []
        requires_grad = False
        if lhs.requires_grad:
            lst.append((lhs, grad_fn_l))
            requires_grad = True
        if rhs.requires_grad:
            lst.append((rhs, grad_fn_r))
            requires_grad = True
        return Tensor(
            data=lhs.data / rhs.data,
            requires_grad=requires_grad,
            depend_on=lst,
        )

    def __rtruediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other / self

    def __matmul__(self, other):
        """
        y=a@b;
        dL/da=dL/dy@b^T;
        dL/db=a^T@dL/dy;
        """
        if not isinstance(other, Tensor):
            other = Tensor(other)
        if len(self.shape) != 2 or len(other.shape) != 2 or self.shape[-1] != other.shape[-2]:
            raise TypeError("This two tensor could not multiply.")

        def grad_fn_l(grad):
            return np.dot(grad, other.data.T)

        def grad_fn_r(grad):
            return np.dot(self.data.T, grad)

        lst = []
        requires_grad = False
        if self.requires_grad:
            lst.append((self, grad_fn_l))
            requires_grad = True
        if other.requires_grad:
            lst.append((other, grad_fn_r))
            requires_grad = True
        return Tensor(
            data=np.dot(self.data, other.data),
            requires_grad=requires_grad,
            depend_on=lst,
        )

    def __rmatmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other @ self

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones(self.shape)
        self.grad += grad
        for tensor, grad_fn in self._depend_on:
            tensor.backward(grad_fn(grad))

    def zero_grad(self):
        self.grad = np.zeros(self.shape)
        for tensor, grad_fn in self._depend_on:
            tensor.zero_grad()
