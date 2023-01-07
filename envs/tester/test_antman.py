#from envs.pacman import AntManEnv
from envs.ant_maze import AntMaze, AntCross, AntFork


env = AntFork()
#env = AntMaze(init_pos=(0, 3), maze_id=4, reset_loc=True)
images = []

env.reset()
images.append(env.render(mode='rgb_array'))

for i in range(10):
    obs = env.step(env.action_space.sample())[0]
    print(obs[..., :2] * 100 * 4)
    images.append(env.render(mode='rgb_array'))

from tools.utils import animate
animate(images, 'output.mp4', _return=False)