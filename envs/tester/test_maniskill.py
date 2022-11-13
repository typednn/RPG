from envs.maniskill_env import make

env = make("Pour-v0")

env.reset()
images = []

images.append(env.render('rgb_array'))

for i in range(10):
    env.step(env.action_space.sample())
    images.append(env.render('rgb_array'))

from tools.utils import animate

animate(images, 'output.mp4')