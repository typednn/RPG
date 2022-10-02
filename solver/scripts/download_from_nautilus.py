import os
import tqdm
from solver import DATA_PATH

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='simple_envhub', type=str)
args = parser.parse_args()


for dataset in args.dataset.split(","):
    data_path = os.path.join(DATA_PATH, dataset)
    os.makedirs(data_path, exist_ok=True)

    for i in tqdm.trange(30):
        os.system(f"kubectl cp hza-try:/cephfs/hza/data/{dataset}/{i} {data_path}/{i}")
