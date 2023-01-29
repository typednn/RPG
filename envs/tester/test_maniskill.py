from envs.maniskill_env import make

env = make()

env.reset()
images = []
print(env.action_space)

images.append(env.render('rgb_array'))
print(env.action_space)

for i in range(10):
    env.step(env.action_space.sample())
    images.append(env.render('rgb_array'))

from tools.utils import animate

animate(images, 'output.mp4')