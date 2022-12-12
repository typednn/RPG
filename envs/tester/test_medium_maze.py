from envs.maze import MediumMaze
import matplotlib.pyplot as plt

env = MediumMaze()

img = env.render('rgb_array')

plt.imshow(img)
plt.savefig('test.png')