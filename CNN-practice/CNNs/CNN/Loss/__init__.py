from .MSE import MSE
from .CrossEntropySoftmax import CrossEntropySoftmax

loss_dict = {
    "MSE": MSE(),
    "cross_entropy_softmax": CrossEntropySoftmax(),
}
