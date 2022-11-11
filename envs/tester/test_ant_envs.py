from envs.ant_envs import AntHEnv

env = AntHEnv("AntPush")
#env = gym.make("HalfCheetah-v3")
images = []

env.reset()
images.append(env.render(mode='rgb_array'))

for i in range(10):
    env.step(env.action_space.sample())
    images.append(env.render(mode='rgb_array'))

from tools.utils import animate
animate(images, 'output.gif', _return=False)