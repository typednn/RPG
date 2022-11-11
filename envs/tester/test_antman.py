from envs.pacman import AntManEnv


env = AntManEnv()
images = []

env.reset()
images.append(env.render(mode='rgb_array'))

for i in range(10):
    env.step(env.action_space.sample())
    images.append(env.render(mode='rgb_array'))

from tools.utils import animate
animate(images, 'output.gif', _return=False)