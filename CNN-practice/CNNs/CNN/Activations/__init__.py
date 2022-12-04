from .Relu import Relu
from .Sigmoid import Sigmoid
from .LeakyRelu_1e_1 import LeakyRelu_1e_1
from .LeakyRelu_1e_2 import LeakyRelu_1e_2
from .Tanh import Tanh

activation_dict = {
    'relu': Relu(),
    'sigmoid': Sigmoid(),
    'leaky_relu_1e-1': LeakyRelu_1e_1(),
    "leaky_relu_1e-2": LeakyRelu_1e_2(),
    "tanh": Tanh(),
}
