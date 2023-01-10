import numpy as np
import matplotlib.pyplot as plt
import tqdm
from envs.maniskill.ee_arm import EEArm


env = EEArm()
env.reset()
print(env.action_space.shape)

images = []
trajs = []
history = None
for i in tqdm.trange(40000):
    obs = env.step(env.action_space.sample())[0]
    trajs.append(obs)
    if (i + 1) % 100 == 0:
        data = env._render_traj_rgb({'next_obs': np.array(trajs)}, occ_val=1, history=history)
        history = data['history']

        print(data['metric'])
        images.append(data['image']['traj'])
        plt.clf()
        plt.imshow(data['image']['traj'])
        plt.savefig('output.png')

        trajs = []

from tools.utils import animate
animate(images, 'output.mp4')