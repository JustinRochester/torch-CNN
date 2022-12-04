from .Adam import AdamOptimizer
from .SGD import SGDOptimizer

optimizer_dict = {
    'Adam': AdamOptimizer,
    'SGD': SGDOptimizer,
}