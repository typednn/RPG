import os
import tqdm
from solver import DATA_PATH


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='simple_envhub', type=str)
args = parser.parse_args()

for name in args.dataset.split(","):
    #name = 'simple_envhub'
    for i in ['train.h5', 'train_scene.pkl', 'val.h5', 'val_scene.pkl']:
        i = f'{name}/{i}'
        os.system(f"kubectl cp {DATA_PATH}/{i} hza-try:/cephfs/hza/data/{i}")