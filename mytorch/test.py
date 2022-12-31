import nptorch
import nptorch.functional as F

x = nptorch.Tensor(
    data=[[-1, 2, -3]],
    requires_grad=True,
)
a = nptorch.Tensor(
    data=[[9], [-8], [7]],
    requires_grad=True,
)
y1 = a @ x
print(y1.data)
y2 = y1 / 10
y3 = F.tanh(y2)
print(y3.data)
y3.backward()
print(y1.grad)
# y2 = y1 * y1
# y2.backward()
print(x.grad)
print(a.grad)
