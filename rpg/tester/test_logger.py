import numpy as np
from tools.utils import logger

logger.configure()
#y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
y = np.random.random((1000,))
print(y.std())

for i in range(10):
    logger.logkv_mean_std('test', y[i])

logger.dumpkvs()