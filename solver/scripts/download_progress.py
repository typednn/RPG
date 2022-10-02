import sys
import os
import tqdm
from solver import MODEL_PATH

model = sys.argv[1].split(',')

os.makedirs(os.path.join(MODEL_PATH, 'progress'), exist_ok=True)

for i in model:
    os.system(f"kubectl cp hza-try:/cephfs/hza/models/{i}/progress.csv {MODEL_PATH}/progress/{i}.csv")
