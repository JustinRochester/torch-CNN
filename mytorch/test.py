import nptorch

x = nptorch.Tensor(
    data=[1, 2, 3],
    requires_grad=True,
)
a = nptorch.Tensor(
    data=[[9], [8], [7]],
    requires_grad=True,
)
y1 = a / x
y1.backward()
# y2 = y1 * y1
# y2.backward()
print(x.grad)
print(a.grad)
