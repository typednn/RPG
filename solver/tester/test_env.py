from solver.engine import Engine
from solver.envs import Move

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--env", type=str, default="EnvHub")
parser.add_argument("--episode", type=int, default=1)

args = parser.parse_args()


engine = Engine(
    env=dict(
        TYPE=args.env,
    ),
)
env = engine.env

from diffrl.utils import animate
demo = engine.make_demo(episode=args.episode)
animate(demo)