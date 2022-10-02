import os
from cpt.solver import make_env
from cpt.solver import dist_utils

def main():
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    env = make_env()
    print(dist_utils.get_rank(), '/', dist_utils.get_world_size())

if __name__ == '__main__':
    dist_utils.launch(main)