from tools.utils import animate
from envs.mujoco_env import make
import tqdm

env = make('HalfCheetah-v3')


print(env.observation_space.shape)
env.reset()
outs = []
for i in tqdm.trange(1000):
    obs, _, done, info = env.step(env.action_space.sample())
    outs.append(obs[-3:].transpose(1, 2, 0))
#print(done, info)
animate(outs)