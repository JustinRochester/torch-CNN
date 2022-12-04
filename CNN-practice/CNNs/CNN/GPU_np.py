use_GPU = 1

if use_GPU:
    import cupy as np
else:
    import numpy as np


if __name__ == '__main__':
    print(np)
