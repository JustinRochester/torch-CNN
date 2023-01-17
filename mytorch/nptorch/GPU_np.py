use_gpu = True

if use_gpu:
    import cupy as np
else:
    import numpy as np
