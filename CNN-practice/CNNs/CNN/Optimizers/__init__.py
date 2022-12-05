from .Adam import AdamOptimizer
from .SGD import SGDOptimizer
from .Memory import MemoryOptimizer

optimizer_dict = {
    'Adam': AdamOptimizer,
    'SGD': SGDOptimizer,
    'Memory': MemoryOptimizer,
}
