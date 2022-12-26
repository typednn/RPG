#from envs.pacman import AntManEnv
from envs.ant_maze import AntMaze


env = AntMaze(init_pos=(1, 1), maze_id=4)
images = []

env.reset()
images.append(env.render(mode='rgb_array'))

for i in range(10):
    env.step(env.action_space.sample())
    images.append(env.render(mode='rgb_array'))

from tools.utils import animate
animate(images, 'output.mp4', _return=False)