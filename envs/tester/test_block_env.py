from envs.block import BlockEnv

env = BlockEnv()
env.reset()
images = []

images.append(env.render('rgb_array'))

for i in range(500):
    env.step(env.action_space.sample())
    images.append(env.render('rgb_array'))

from tools.utils import animate

animate(images, 'output.mp4', _return=False)