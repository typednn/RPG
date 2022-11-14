from envs.fetch.sequential import SequentialStack

env = SequentialStack()
env.reset()
images = []


for i in range(3):
    env.reset()
    images.append(env.render('rgb_array'))

    for j in range(100):
        env.step(env.action_space.sample())
        images.append(env.render('rgb_array'))

from tools.utils import animate

animate(images, 'output.gif')