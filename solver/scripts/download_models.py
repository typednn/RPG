import sys
import os
import tqdm
from solver import MODEL_PATH

model = sys.argv[1].split(',')

for i in model:
    os.system(f"kubectl cp hza-try:/cephfs/hza/models/{i} {MODEL_PATH}/{i}")
